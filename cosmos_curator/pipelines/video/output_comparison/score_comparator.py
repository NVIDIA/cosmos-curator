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
"""Motion and aesthetic score comparison for split pipeline outputs.

This module is longer than the score comparison math because it owns the full
feature lifecycle for Ray Data execution:

1. ``ScoreFeatureComparator`` builds one clip feature plan for either
   ``aesthetic_score`` or ``motion_score``. When both score features are
   enabled, the generic feature pipeline groups their matching
   ``ClipArtifactsLoadWorker`` inputs so metadata is still loaded once per clip.
2. ``ScoreClipCompareResult`` is the compact JSON-compatible row shape emitted
   by the per-clip compare stage. Ray Data rows cross worker boundaries as
   dictionaries, so the module keeps explicit encode/decode validation close to
   the feature.
3. ``compare_score_clip_artifacts`` normalizes loaded metadata into score
   observations, records missing/invalid score fields, and compares only valid
   numeric values with the configured tolerances for that score feature.
4. ``reduce_score_clip_results`` aggregates one score feature's clip rows into
   feature-level issues, status, and metrics for the final comparison report.

The small core comparison is intentionally surrounded by structured reporting
logic so partial runs are actionable: a run may have motion scores, aesthetic
scores, both, or neither, and malformed score fields should be reported
separately from value mismatches.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from math import isclose, isfinite
from typing import Self, cast

import attrs

from cosmos_curator.pipelines.video.output_comparison.feature_plan import (
    ClipFeaturePlan,
    FeatureComparisonContext,
    FeatureComparisonPlan,
    FeatureComparisonResult,
    ResolvedFeaturePlan,
)
from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue
from cosmos_curator.pipelines.video.output_comparison.report import FeatureComparison, FeatureComparisonStatus, Issue
from cosmos_curator.pipelines.video.output_comparison.video_artifacts import (
    ClipArtifactsLoadWorker,
    LoadedClipArtifacts,
)
from cosmos_curator.pipelines.video.output_comparison.video_planning import build_clip_comparison_specs

MOTION_SCORE_FEATURE_NAME = "motion_score"
AESTHETIC_SCORE_FEATURE_NAME = "aesthetic_score"
SCORE_FEATURE_NAMES = (AESTHETIC_SCORE_FEATURE_NAME, MOTION_SCORE_FEATURE_NAME)
_MOTION_SCORE_FIELDS = ("motion_score.global_mean", "motion_score.per_patch_min_256")
_AESTHETIC_SCORE_FIELDS = ("aesthetic_score",)
_SCORE_FIELDS_BY_FEATURE = {
    AESTHETIC_SCORE_FEATURE_NAME: _AESTHETIC_SCORE_FIELDS,
    MOTION_SCORE_FEATURE_NAME: _MOTION_SCORE_FIELDS,
}
_MISSING = object()


def _non_negative_finite_tolerance(_instance: object, attribute: "attrs.Attribute[float]", value: float) -> None:
    if not isfinite(value) or value < 0:
        error_msg = f"{attribute.name} must be a finite number greater than or equal to 0"
        raise ValueError(error_msg)


def _score_feature_name_validator(_instance: object, attribute: "attrs.Attribute[str]", value: str) -> None:
    try:
        _validate_score_feature_name(value)
    except ValueError as exc:
        error_msg = f"{attribute.name} must be one of {sorted(SCORE_FEATURE_NAMES)}"
        raise ValueError(error_msg) from exc


def _validate_score_feature_name(feature_name: str) -> None:
    if feature_name not in SCORE_FEATURE_NAMES:
        error_msg = f"unknown score feature: {feature_name}"
        raise ValueError(error_msg)


@attrs.define(frozen=True)
class ScoreComparisonPolicy:
    """Policy for one per-clip score feature comparison."""

    metadata_version: str = "v0"
    abs_tolerance: float = attrs.field(default=1e-6, validator=_non_negative_finite_tolerance)
    rel_tolerance: float = attrs.field(default=1e-6, validator=_non_negative_finite_tolerance)


DEFAULT_SCORE_POLICY = ScoreComparisonPolicy()


@attrs.define(frozen=True)
class _ScoreObservation:
    """One output side's normalized score data for one feature."""

    present: bool
    values: Mapping[str, float] = attrs.field(factory=dict)
    invalid_fields: Mapping[str, str] = attrs.field(factory=dict)


@attrs.define(frozen=True)
class ScoreComparisonCounts:
    """Counters for one score feature comparison."""

    clips_with_scores_a: int = 0
    clips_with_scores_b: int = 0
    clips_compared: int = 0
    fields_compared: int = 0

    def to_json_dict(self, *, policy: ScoreComparisonPolicy | None = None) -> JsonDictObject:
        """Convert counters and optional policy settings to JSON-compatible metrics."""
        metrics = cast("JsonDictObject", attrs.asdict(self))
        if policy is not None:
            metrics["score_abs_tolerance"] = policy.abs_tolerance
            metrics["score_rel_tolerance"] = policy.rel_tolerance
        return metrics

    @classmethod
    def from_json_dict(cls, row: Mapping[str, JsonValue]) -> Self:
        """Build counters from a Ray Data row."""
        return cls(
            clips_with_scores_a=_required_int(row, "clips_with_scores_a"),
            clips_with_scores_b=_required_int(row, "clips_with_scores_b"),
            clips_compared=_required_int(row, "clips_compared"),
            fields_compared=_required_int(row, "fields_compared"),
        )


@attrs.define(frozen=True)
class ScoreClipCompareResult:
    """Compact score comparison result emitted for one clip."""

    video_key: str
    clip_id: str
    issues: tuple[Issue, ...]
    counts: ScoreComparisonCounts

    def to_json_dict(self) -> JsonDictObject:
        """Convert the result to a JSON-compatible Ray Data row."""
        return {
            "video_key": self.video_key,
            "clip_id": self.clip_id,
            "issues": [issue.to_json_dict() for issue in self.issues],
            "counts": self.counts.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, row: Mapping[str, JsonValue]) -> Self:
        """Build a compact score comparison result from a Ray Data row."""
        issues_value = row["issues"]
        counts_value = row["counts"]
        if not isinstance(issues_value, list):
            error_msg = "score clip result row field 'issues' must be a list"
            raise TypeError(error_msg)
        if not isinstance(counts_value, dict):
            error_msg = "score clip result row field 'counts' must be an object"
            raise TypeError(error_msg)
        return cls(
            video_key=_required_str(row, "video_key"),
            clip_id=_required_str(row, "clip_id"),
            issues=tuple(Issue.from_json_dict(issue) for issue in issues_value),
            counts=ScoreComparisonCounts.from_json_dict(counts_value),
        )


@attrs.define(frozen=True)
class ScoreFeatureComparator:
    """Compare one per-clip score feature using loaded metadata."""

    feature_name: str = attrs.field(validator=_score_feature_name_validator)
    policy: ScoreComparisonPolicy = DEFAULT_SCORE_POLICY

    @property
    def name(self) -> str:
        """Return the planner name used for logging."""
        return self.feature_name

    def build_plan(self, context: FeatureComparisonContext) -> FeatureComparisonPlan:
        """Build a clip plan for this score feature comparison."""
        clip_specs = build_clip_comparison_specs(context.specs)
        if not clip_specs:
            return ResolvedFeaturePlan(self.name, _empty_score_feature_result(self.policy))
        return ClipFeaturePlan(
            feature_name=self.name,
            clip_specs=clip_specs,
            load_worker_class=ClipArtifactsLoadWorker,
            load_worker_constructor_kwargs={
                "profile_name": context.profile_name,
                "metadata_version": self.policy.metadata_version,
            },
            compare_row=cast("Callable[[Mapping[str, JsonValue]], JsonDictObject]", self._compare_clip_row),
            reduce_rows=cast(
                "Callable[[Sequence[Mapping[str, JsonValue]]], FeatureComparisonResult]",
                self._reduce_clip_rows,
            ),
        )

    def _compare_clip_row(self, row: Mapping[str, JsonValue]) -> JsonDictObject:
        """Compare one loaded clip artifact row."""
        return compare_score_clip_artifacts(
            LoadedClipArtifacts.from_json_dict(row),
            self.feature_name,
            self.policy,
        ).to_json_dict()

    def _reduce_clip_rows(self, rows: Sequence[Mapping[str, JsonValue]]) -> FeatureComparisonResult:
        """Reduce compact per-clip score rows into feature results."""
        return reduce_score_clip_results(
            tuple(
                sorted(
                    (ScoreClipCompareResult.from_json_dict(row) for row in rows),
                    key=lambda result: (result.video_key, result.clip_id),
                )
            ),
            feature_name=self.feature_name,
            policy=self.policy,
        )


def compare_score_clip_artifacts(
    artifacts: LoadedClipArtifacts,
    feature_name: str,
    policy: ScoreComparisonPolicy,
) -> ScoreClipCompareResult:
    """Compare score metadata for one loaded clip artifact row."""
    _validate_score_feature_name(feature_name)
    observations_a = _score_observations(artifacts.metadata_a)
    observations_b = _score_observations(artifacts.metadata_b)
    issues, counts = _compare_score_feature(
        artifacts,
        feature_name,
        observations_a[feature_name],
        observations_b[feature_name],
        policy=policy,
    )
    return ScoreClipCompareResult(
        video_key=artifacts.spec.video_key,
        clip_id=artifacts.spec.clip_id,
        issues=tuple(issues),
        counts=counts,
    )


def reduce_score_clip_results(
    clip_results: Sequence[ScoreClipCompareResult],
    *,
    feature_name: str,
    policy: ScoreComparisonPolicy,
) -> FeatureComparisonResult:
    """Reduce one score feature's clip comparison rows into a feature result."""
    _validate_score_feature_name(feature_name)
    counts = _sum_counts(result.counts for result in clip_results)
    issues = tuple(issue for result in clip_results for issue in result.issues)
    status = _score_status(counts, issues)
    return FeatureComparisonResult(
        issues=issues,
        comparison=FeatureComparison(
            status=status,
            metrics=counts.to_json_dict(policy=policy),
        ),
    )


def _empty_score_feature_result(policy: ScoreComparisonPolicy) -> FeatureComparisonResult:
    return FeatureComparisonResult(
        issues=(),
        comparison=FeatureComparison(
            status="skipped",
            metrics=ScoreComparisonCounts().to_json_dict(policy=policy),
        ),
    )


def _compare_score_feature(
    artifacts: LoadedClipArtifacts,
    feature_name: str,
    observation_a: _ScoreObservation,
    observation_b: _ScoreObservation,
    *,
    policy: ScoreComparisonPolicy,
) -> tuple[list[Issue], ScoreComparisonCounts]:
    issues: list[Issue] = []
    fields_compared = 0
    issues.extend(_score_metadata_unavailable_issues(artifacts, feature_name, observation_a, observation_b))
    if artifacts.spec.in_a and artifacts.spec.in_b:
        issues.extend(_score_invalid_issues(artifacts, "a", feature_name, observation_a))
        issues.extend(_score_invalid_issues(artifacts, "b", feature_name, observation_b))
        issues.extend(_score_presence_issues(artifacts, feature_name, observation_a, observation_b))
        if observation_a.present and observation_b.present:
            value_issues, fields_compared = _score_value_issues(
                artifacts,
                feature_name,
                observation_a,
                observation_b,
                policy=policy,
            )
            issues.extend(value_issues)
    counts = ScoreComparisonCounts(
        clips_with_scores_a=1 if artifacts.spec.in_a and _complete_valid_score(feature_name, observation_a) else 0,
        clips_with_scores_b=1 if artifacts.spec.in_b and _complete_valid_score(feature_name, observation_b) else 0,
        clips_compared=1
        if artifacts.spec.in_a
        and artifacts.spec.in_b
        and _complete_valid_score(feature_name, observation_a)
        and _complete_valid_score(feature_name, observation_b)
        else 0,
        fields_compared=fields_compared,
    )
    return issues, counts


def _score_observations(metadata: JsonDictObject | None) -> dict[str, _ScoreObservation]:
    if metadata is None:
        return {
            AESTHETIC_SCORE_FEATURE_NAME: _ScoreObservation(present=False),
            MOTION_SCORE_FEATURE_NAME: _ScoreObservation(present=False),
        }
    return {
        AESTHETIC_SCORE_FEATURE_NAME: _aesthetic_score_observation(metadata),
        MOTION_SCORE_FEATURE_NAME: _motion_score_observation(metadata),
    }


def _aesthetic_score_observation(metadata: JsonDictObject) -> _ScoreObservation:
    value = metadata.get(AESTHETIC_SCORE_FEATURE_NAME, _MISSING)
    if value is _MISSING:
        return _ScoreObservation(present=False)
    numeric_value = _numeric_value(value)
    if numeric_value is None:
        return _ScoreObservation(
            present=True,
            invalid_fields={AESTHETIC_SCORE_FEATURE_NAME: "must be numeric"},
        )
    return _ScoreObservation(present=True, values={AESTHETIC_SCORE_FEATURE_NAME: numeric_value})


def _motion_score_observation(metadata: JsonDictObject) -> _ScoreObservation:
    value = metadata.get(MOTION_SCORE_FEATURE_NAME, _MISSING)
    if value is _MISSING:
        return _ScoreObservation(present=False)
    if not isinstance(value, dict):
        return _ScoreObservation(
            present=True,
            invalid_fields={MOTION_SCORE_FEATURE_NAME: "must be an object"},
        )
    values: dict[str, float] = {}
    invalid_fields: dict[str, str] = {}
    for nested_field in ("global_mean", "per_patch_min_256"):
        field = f"{MOTION_SCORE_FEATURE_NAME}.{nested_field}"
        nested_value = value.get(nested_field, _MISSING)
        if nested_value is _MISSING:
            invalid_fields[field] = "is required"
            continue
        numeric_value = _numeric_value(nested_value)
        if numeric_value is None:
            invalid_fields[field] = "must be numeric"
        else:
            values[field] = numeric_value
    return _ScoreObservation(present=True, values=values, invalid_fields=invalid_fields)


def _score_presence_issues(
    artifacts: LoadedClipArtifacts,
    feature_name: str,
    observation_a: _ScoreObservation,
    observation_b: _ScoreObservation,
) -> list[Issue]:
    if observation_a.present == observation_b.present:
        return []
    missing_output = "a" if observation_b.present else "b"
    details: JsonDictObject = {
        "a_present": observation_a.present,
        "b_present": observation_b.present,
    }
    if _metadata_was_unavailable(artifacts, missing_output):
        return []
    return [
        Issue(
            code=f"{feature_name}_field_missing",
            message="Score field is present on only one output",
            output=missing_output,
            feature=feature_name,
            field=feature_name,
            video=artifacts.spec.video_key,
            clip=artifacts.spec.clip_id,
            details=details,
        )
    ]


def _score_metadata_unavailable_issues(
    artifacts: LoadedClipArtifacts,
    feature_name: str,
    observation_a: _ScoreObservation,
    observation_b: _ScoreObservation,
) -> list[Issue]:
    """Report unavailable metadata needed to inspect a score feature."""
    issues: list[Issue] = []
    for output_label in ("a", "b"):
        counterpart_has_score = observation_b.present if output_label == "a" else observation_a.present
        details = _metadata_unavailable_details(
            artifacts,
            output_label,
            report_missing=counterpart_has_score,
        )
        if details is None:
            continue
        issues.append(
            Issue(
                code=f"{feature_name}_metadata_unavailable",
                message="Score metadata could not be inspected",
                output=output_label,
                feature=feature_name,
                field=feature_name,
                video=artifacts.spec.video_key,
                clip=artifacts.spec.clip_id,
                details=details,
            )
        )
    return issues


def _metadata_unavailable_details(
    artifacts: LoadedClipArtifacts,
    output_label: str,
    *,
    report_missing: bool,
) -> JsonDictObject | None:
    if output_label == "a":
        if not artifacts.spec.in_a:
            return None
        missing_metadata = artifacts.missing_metadata_a
        invalid_metadata = artifacts.invalid_metadata_a
    else:
        if not artifacts.spec.in_b:
            return None
        missing_metadata = artifacts.missing_metadata_b
        invalid_metadata = artifacts.invalid_metadata_b

    metadata_path = _metadata_path_for_output(artifacts, output_label)
    if missing_metadata and report_missing:
        details: JsonDictObject = {"reason": "missing"}
    elif invalid_metadata is not None:
        details = {"reason": "invalid", "error": invalid_metadata}
    else:
        return None
    if metadata_path is not None:
        details["metadata_path"] = metadata_path
    return details


def _score_invalid_issues(
    artifacts: LoadedClipArtifacts,
    output_label: str,
    feature_name: str,
    observation: _ScoreObservation,
) -> list[Issue]:
    issues: list[Issue] = []
    metadata_path = _metadata_path_for_output(artifacts, output_label)
    for field, reason in sorted(observation.invalid_fields.items()):
        details: JsonDictObject = {"reason": reason}
        if metadata_path is not None:
            details["metadata_path"] = metadata_path
        issues.append(
            Issue(
                code=f"{feature_name}_field_invalid",
                message="Score field has an invalid shape",
                output=output_label,
                feature=feature_name,
                field=field,
                video=artifacts.spec.video_key,
                clip=artifacts.spec.clip_id,
                details=details,
            )
        )
    return issues


def _score_value_issues(
    artifacts: LoadedClipArtifacts,
    feature_name: str,
    observation_a: _ScoreObservation,
    observation_b: _ScoreObservation,
    *,
    policy: ScoreComparisonPolicy,
) -> tuple[list[Issue], int]:
    issues: list[Issue] = []
    fields_compared = 0
    abs_tolerance = policy.abs_tolerance
    rel_tolerance = policy.rel_tolerance
    for field in _SCORE_FIELDS_BY_FEATURE[feature_name]:
        if field in observation_a.invalid_fields or field in observation_b.invalid_fields:
            continue
        value_a = observation_a.values.get(field)
        value_b = observation_b.values.get(field)
        if value_a is None or value_b is None:
            continue
        fields_compared += 1
        abs_delta = abs(value_a - value_b)
        rel_delta = _relative_delta(value_a, value_b, abs_delta)
        if isclose(value_a, value_b, rel_tol=rel_tolerance, abs_tol=abs_tolerance):
            continue
        issues.append(
            Issue(
                code=f"{feature_name}_value_mismatch",
                message="Score field differs beyond configured tolerance",
                feature=feature_name,
                field=field,
                video=artifacts.spec.video_key,
                clip=artifacts.spec.clip_id,
                details={
                    "a": value_a,
                    "b": value_b,
                    "abs_delta": abs_delta,
                    "rel_delta": rel_delta,
                    "score_abs_tolerance": abs_tolerance,
                    "score_rel_tolerance": rel_tolerance,
                },
            )
        )
    return issues, fields_compared


def _complete_valid_score(feature_name: str, observation: _ScoreObservation) -> bool:
    return (
        observation.present
        and not observation.invalid_fields
        and all(field in observation.values for field in _SCORE_FIELDS_BY_FEATURE[feature_name])
    )


def _sum_counts(counts: Iterable[ScoreComparisonCounts]) -> ScoreComparisonCounts:
    counts = tuple(counts)
    return ScoreComparisonCounts(
        clips_with_scores_a=sum(count.clips_with_scores_a for count in counts),
        clips_with_scores_b=sum(count.clips_with_scores_b for count in counts),
        clips_compared=sum(count.clips_compared for count in counts),
        fields_compared=sum(count.fields_compared for count in counts),
    )


def _score_status(counts: ScoreComparisonCounts, issues: Sequence[Issue]) -> FeatureComparisonStatus:
    if issues:
        return "failed"
    if counts.clips_with_scores_a == 0 and counts.clips_with_scores_b == 0:
        return "skipped"
    return "passed"


def _metadata_was_unavailable(artifacts: LoadedClipArtifacts, output_label: str) -> bool:
    if output_label == "a":
        return artifacts.missing_metadata_a or artifacts.invalid_metadata_a is not None
    return artifacts.missing_metadata_b or artifacts.invalid_metadata_b is not None


def _metadata_path_for_output(artifacts: LoadedClipArtifacts, output_label: str) -> str | None:
    if output_label == "a":
        return artifacts.metadata_path_a
    return artifacts.metadata_path_b


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric_value = float(value)
    if not isfinite(numeric_value):
        return None
    return numeric_value


def _relative_delta(value_a: float, value_b: float, abs_delta: float) -> float:
    denominator = max(abs(value_a), abs(value_b))
    if denominator == 0:
        return 0.0
    return abs_delta / denominator


def _required_int(row: Mapping[str, JsonValue], field: str) -> int:
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, int):
        error_msg = f"score comparison row field {field!r} must be an integer"
        raise TypeError(error_msg)
    return value


def _required_str(row: Mapping[str, JsonValue], field: str) -> str:
    value = row[field]
    if not isinstance(value, str):
        error_msg = f"score comparison row field {field!r} must be a string"
        raise TypeError(error_msg)
    return value
