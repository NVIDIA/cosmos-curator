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
"""Stage output comparison helpers."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import attrs
import numpy as np
import numpy.typing as npt
import smart_open  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, PipelineTask
from cosmos_curator.core.utils.misc.stage_replay import (
    DirectStageExecutor,
    PickleTaskSerializer,
    StageExecutor,
    TaskPath,
    TaskSerializer,
)
from cosmos_curator.core.utils.storage import storage_utils
from cosmos_xenna.pipelines.private.resources import NodeInfo, WorkerMetadata


class TaskComparator(Protocol):
    """Protocol for comparing two pipeline tasks."""

    def checked_fields(self, golden: PipelineTask) -> tuple[str, ...]:
        """Return the leaf field paths this comparator evaluates."""
        ...  # pragma: no cover

    def compare(
        self,
        golden: PipelineTask,
        candidate: PipelineTask,
        *,
        atol: float,
    ) -> list["FieldDiff"]:
        """Compare tasks and return a list of field-level failures."""
        ...  # pragma: no cover


@attrs.define(frozen=True)
class FieldDiff:
    """A single field-level comparison failure."""

    field: str
    detail: str
    max_diff_observed: float | None = None
    shape_mismatch: bool = False


@attrs.define(frozen=True)
class TaskDiff:
    """Comparison result for one task within a batch."""

    batch_file: str
    task_index: int
    checked_fields: tuple[str, ...] = ()
    failures: tuple[FieldDiff, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether the compared task passed."""
        return len(self.failures) == 0


@attrs.define(frozen=True)
class FieldCompareSummary:
    """Aggregated results for one field path."""

    passed: int = 0
    failed: int = 0
    max_diff_observed: float | None = None
    shape_mismatches: int = 0


@attrs.define(frozen=True)
class CompareReport:
    """Aggregated output comparison report."""

    stage: str
    atol: float
    total_batches: int
    passed_batches: int
    failed_batches: int
    pass_rate: float
    fields: dict[str, FieldCompareSummary]
    failures: list[TaskDiff]
    profile_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return {
            "stage": self.stage,
            "atol": self.atol,
            "total_batches": self.total_batches,
            "passed_batches": self.passed_batches,
            "failed_batches": self.failed_batches,
            "pass_rate": self.pass_rate,
            "fields": {
                name: {
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "max_diff_observed": summary.max_diff_observed,
                    "shape_mismatches": summary.shape_mismatches,
                }
                for name, summary in self.fields.items()
            },
            "failures": [
                {
                    "batch_file": task_diff.batch_file,
                    "task_index": task_diff.task_index,
                    "field": failure.field,
                    "detail": failure.detail,
                }
                for task_diff in self.failures
                for failure in task_diff.failures
            ],
        }

    def write_json(self, path: TaskPath) -> None:
        """Write the report to disk as JSON."""
        if isinstance(path, Path):
            storage_utils.create_path(str(path.parent))
        client = (
            storage_utils.get_storage_client(str(path), profile_name=self.profile_name)
            if self.profile_name is not None
            else storage_utils.get_storage_client(str(path))
        )
        client_params = storage_utils.get_smart_open_client_params(client) if client is not None else {}
        with smart_open.open(str(path), "w", encoding="utf-8", **client_params) as f:
            json.dump(self.to_dict(), f, indent=2)


@attrs.define(frozen=True)
class StageCompareResult:
    """Result of a stage compare run."""

    report: CompareReport
    report_path: TaskPath
    pass_threshold: float

    @property
    def passed(self) -> bool:
        """Return whether the run passed the configured threshold."""
        return self.report.pass_rate >= self.pass_threshold


_registry: dict[type[PipelineTask], TaskComparator] = {}


def register_comparator(task_type: type[PipelineTask], comparator: TaskComparator) -> None:
    """Register a comparator for a task type."""
    _registry[task_type] = comparator


def get_comparator(task_type: type[PipelineTask]) -> TaskComparator:
    """Return the comparator for a task type."""
    return _registry.get(task_type, _GenericComparator())


def _compare_arrays(
    field_path: str,
    golden: npt.NDArray[Any],
    candidate: npt.NDArray[Any],
    *,
    atol: float,
) -> list[FieldDiff]:
    """Compare NumPy arrays."""
    if golden.shape != candidate.shape:
        return [
            FieldDiff(
                field=field_path,
                detail=f"shape mismatch golden={golden.shape} new={candidate.shape}",
                shape_mismatch=True,
            )
        ]

    if np.issubdtype(golden.dtype, np.number) and np.issubdtype(candidate.dtype, np.number):
        if np.allclose(golden, candidate, atol=atol, rtol=0.0, equal_nan=True):
            return []
        golden_float = golden.astype(np.float64)
        candidate_float = candidate.astype(np.float64)
        both_nan_mask = np.isnan(golden_float) & np.isnan(candidate_float)
        diff = np.abs(golden_float - candidate_float)
        comparable_diff = diff[~both_nan_mask]
        max_diff = float(np.nanmax(comparable_diff)) if comparable_diff.size > 0 else 0.0
        return [FieldDiff(field=field_path, detail=f"max diff {max_diff}", max_diff_observed=max_diff)]

    if np.array_equal(golden, candidate):
        return []
    return [FieldDiff(field=field_path, detail="array values differ")]


def _compare_attrs(
    field_path: str,
    golden: object,
    candidate: object,
    *,
    atol: float,
) -> list[FieldDiff]:
    failures: list[FieldDiff] = []
    golden_attrs = cast("attrs.AttrsInstance", golden)
    for field in attrs.fields(golden_attrs.__class__):
        child_path = f"{field_path}.{field.name}" if field_path else field.name
        failures.extend(
            _compare_values(
                child_path,
                getattr(golden_attrs, field.name),
                getattr(candidate, field.name),
                atol=atol,
            )
        )
    return failures


def _compare_mapping(
    field_path: str,
    golden: Mapping[object, object],
    candidate: Mapping[object, object],
    *,
    atol: float,
) -> list[FieldDiff]:
    golden_keys = set(golden.keys())
    candidate_keys = set(candidate.keys())
    if golden_keys != candidate_keys:
        return [
            FieldDiff(
                field=field_path,
                detail=(
                    f"dict key mismatch golden={sorted(golden_keys, key=repr)!r} "
                    f"new={sorted(candidate_keys, key=repr)!r}"
                ),
            )
        ]
    failures: list[FieldDiff] = []
    for key in sorted(golden_keys, key=repr):
        child_path = f"{field_path}.{key}" if field_path else str(key)
        failures.extend(_compare_values(child_path, golden[key], candidate[key], atol=atol))
    return failures


def _compare_sequence(
    field_path: str,
    golden: Sequence[object],
    candidate: Sequence[object],
    *,
    atol: float,
) -> list[FieldDiff]:
    if len(golden) != len(candidate):
        return [FieldDiff(field=field_path, detail=f"length mismatch golden={len(golden)} new={len(candidate)}")]
    failures: list[FieldDiff] = []
    for index, (golden_item, candidate_item) in enumerate(zip(golden, candidate, strict=True)):
        failures.extend(_compare_values(f"{field_path}[{index}]", golden_item, candidate_item, atol=atol))
    return failures


def _compare_values(
    field_path: str,
    golden: object,
    candidate: object,
    *,
    atol: float,
) -> list[FieldDiff]:
    """Recursively compare values for the generic comparator."""
    if type(golden) is not type(candidate):
        return [
            FieldDiff(
                field=field_path,
                detail=f"type mismatch golden={type(golden).__name__} new={type(candidate).__name__}",
            )
        ]
    if isinstance(golden, np.ndarray):
        return _compare_arrays(field_path, golden, cast("npt.NDArray[Any]", candidate), atol=atol)
    if attrs.has(golden.__class__):
        return _compare_attrs(field_path, golden, candidate, atol=atol)
    if isinstance(golden, Mapping):
        golden_m = cast("Mapping[object, object]", golden)
        return _compare_mapping(field_path, golden_m, cast("Mapping[object, object]", candidate), atol=atol)
    if isinstance(golden, Sequence) and not isinstance(golden, (str, bytes, bytearray)):
        golden_s = cast("Sequence[object]", golden)
        return _compare_sequence(field_path, golden_s, cast("Sequence[object]", candidate), atol=atol)
    return (
        []
        if golden == candidate
        else [FieldDiff(field=field_path, detail=f"value mismatch golden={golden!r} new={candidate!r}")]
    )


def _collect_attrs_paths(field_path: str, value: object) -> set[str]:
    paths: set[str] = set()
    value_attrs = cast("attrs.AttrsInstance", value)
    for field in attrs.fields(value_attrs.__class__):
        child_path = f"{field_path}.{field.name}" if field_path else field.name
        paths.update(_collect_field_paths(child_path, getattr(value_attrs, field.name)))
    return paths


def _collect_mapping_paths(field_path: str, value: Mapping[object, object]) -> set[str]:
    if len(value) == 0:
        return {field_path}
    paths: set[str] = set()
    for key in sorted(value.keys(), key=repr):
        child_path = f"{field_path}.{key}" if field_path else str(key)
        paths.update(_collect_field_paths(child_path, value[key]))
    return paths


def _collect_sequence_paths(field_path: str, value: Sequence[object]) -> set[str]:
    if len(value) == 0:
        return {field_path}
    paths: set[str] = set()
    for index, item in enumerate(value):
        paths.update(_collect_field_paths(f"{field_path}[{index}]", item))
    return paths


def _collect_field_paths(field_path: str, value: object) -> set[str]:
    """Collect comparable leaf field paths for a value."""
    if isinstance(value, np.ndarray):
        return {field_path}
    if attrs.has(value.__class__):
        return _collect_attrs_paths(field_path, value)
    if isinstance(value, Mapping):
        return _collect_mapping_paths(field_path, cast("Mapping[object, object]", value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _collect_sequence_paths(field_path, cast("Sequence[object]", value))
    return {field_path}


def compare_attrs_fields(
    golden: object,
    candidate: object,
    *,
    field_names: Sequence[str],
    atol: float,
) -> list[FieldDiff]:
    """Compare a selected set of attrs fields using generic recursion."""
    failures: list[FieldDiff] = []
    for field_name in field_names:
        failures.extend(
            _compare_values(
                field_name,
                getattr(golden, field_name),
                getattr(candidate, field_name),
                atol=atol,
            )
        )
    return failures


def collect_checked_attrs_fields(golden: object, *, field_names: Sequence[str]) -> tuple[str, ...]:
    """Collect leaf field paths for a selected set of attrs fields."""
    checked_fields: set[str] = set()
    for field_name in field_names:
        checked_fields.update(_collect_field_paths(field_name, getattr(golden, field_name)))
    return tuple(sorted(checked_fields))


class _GenericComparator:
    """Generic attrs-aware task comparator."""

    def checked_fields(self, golden: PipelineTask) -> tuple[str, ...]:
        """Return all comparable leaf fields for the task."""
        return tuple(sorted(_collect_field_paths("", golden)))

    def compare(
        self,
        golden: PipelineTask,
        candidate: PipelineTask,
        *,
        atol: float,
    ) -> list[FieldDiff]:
        """Compare two tasks using recursive attrs reflection."""
        return _compare_values("", golden, candidate, atol=atol)


def _summarize_task_diffs(task_diffs: list[TaskDiff]) -> dict[str, FieldCompareSummary]:
    """Aggregate field-level failures across tasks."""
    summaries: dict[str, FieldCompareSummary] = {}
    for task_diff in task_diffs:
        failed_fields = {failure.field for failure in task_diff.failures}
        for checked_field in task_diff.checked_fields:
            summary = summaries.get(checked_field, FieldCompareSummary())
            summaries[checked_field] = FieldCompareSummary(
                passed=summary.passed + int(checked_field not in failed_fields),
                failed=summary.failed + int(checked_field in failed_fields),
                max_diff_observed=summary.max_diff_observed,
                shape_mismatches=summary.shape_mismatches,
            )

        for failure in task_diff.failures:
            summary = summaries.get(failure.field, FieldCompareSummary())
            max_diff_observed = summary.max_diff_observed
            if failure.max_diff_observed is not None:
                max_diff_observed = (
                    failure.max_diff_observed
                    if max_diff_observed is None
                    else max(max_diff_observed, failure.max_diff_observed)
                )
            summaries[failure.field] = FieldCompareSummary(
                passed=summary.passed,
                failed=summary.failed,
                max_diff_observed=max_diff_observed,
                shape_mismatches=summary.shape_mismatches + int(failure.shape_mismatch),
            )

    return summaries


def _compare_task_lists(
    batch_file: str,
    golden_tasks: list[PipelineTask],
    candidate_tasks: list[PipelineTask],
    *,
    atol: float,
) -> list[TaskDiff]:
    """Compare two task lists."""
    if len(golden_tasks) != len(candidate_tasks):
        return [
            TaskDiff(
                batch_file=batch_file,
                task_index=0,
                checked_fields=("tasks",),
                failures=(
                    FieldDiff(
                        field="tasks",
                        detail=f"task count mismatch golden={len(golden_tasks)} new={len(candidate_tasks)}",
                    ),
                ),
            )
        ]

    task_diffs: list[TaskDiff] = []
    for task_index, (golden_task, candidate_task) in enumerate(zip(golden_tasks, candidate_tasks, strict=True)):
        comparator = get_comparator(type(golden_task))
        failures = comparator.compare(golden_task, candidate_task, atol=atol)
        checked_fields = comparator.checked_fields(golden_task)
        task_diffs.append(
            TaskDiff(
                batch_file=batch_file,
                task_index=task_index,
                checked_fields=checked_fields,
                failures=tuple(failures),
            )
        )
    return task_diffs


def run_stage_compare(  # noqa: PLR0913, C901
    stages: list[CuratorStage],
    input_path: TaskPath,
    golden_path: TaskPath,
    atol: float,
    limit: int,
    pass_threshold: float,
    *,
    report_path: TaskPath,
    profile_name: str | None = None,
    executor: StageExecutor | None = None,
    serializer: TaskSerializer | None = None,
) -> StageCompareResult:
    """Run stage comparison from saved input tasks against golden tasks."""
    if len(stages) == 0:
        msg = "No stages to compare"
        raise ValueError(msg)

    _serializer = (
        serializer
        if serializer is not None
        else (PickleTaskSerializer(profile_name=profile_name) if profile_name is not None else PickleTaskSerializer())
    )
    _executor = executor if executor is not None else DirectStageExecutor()

    input_files = _serializer.find_task_files(input_path, "*.pkl", limit)
    golden_files = _serializer.find_task_files(golden_path, "*.pkl", limit)

    if len(input_files) == 0:
        msg = f"No input tasks found in {input_path}"
        raise ValueError(msg)
    if len(golden_files) == 0:
        msg = f"No golden tasks found in {golden_path}"
        raise ValueError(msg)
    if len(input_files) != len(golden_files):
        msg = f"Input/golden batch count mismatch: input={len(input_files)} golden={len(golden_files)}"
        raise ValueError(msg)

    input_files_by_name = {Path(str(path)).name: path for path in input_files}
    golden_files_by_name = {Path(str(path)).name: path for path in golden_files}
    if input_files_by_name.keys() != golden_files_by_name.keys():
        msg = (
            "Input/golden batch file mismatch: "
            f"input={sorted(input_files_by_name)} golden={sorted(golden_files_by_name)}"
        )
        raise ValueError(msg)

    sorted_batch_names = sorted(input_files_by_name)
    input_batches = [_serializer.load(input_files_by_name[name]) for name in sorted_batch_names]
    golden_batches = [_serializer.load(golden_files_by_name[name]) for name in sorted_batch_names]

    node_info, worker_metadata = NodeInfo(node_id="localhost"), WorkerMetadata.make_dummy()
    candidate_batches = input_batches
    for stage in stages:
        candidate_batches = _executor.execute_stage(stage, candidate_batches, node_info, worker_metadata)

    if len(candidate_batches) != len(golden_batches):
        msg = f"Candidate/golden batch count mismatch: candidate={len(candidate_batches)} golden={len(golden_batches)}"
        raise ValueError(msg)

    all_task_diffs: list[TaskDiff] = []
    passed_batches = 0
    total_batches = len(candidate_batches)
    stage_name = stages[-1].__class__.__name__

    for batch_index, (candidate_tasks, golden_tasks, batch_name) in enumerate(
        zip(candidate_batches, golden_batches, sorted_batch_names, strict=True),
        start=1,
    ):
        task_diffs = _compare_task_lists(batch_name, golden_tasks, candidate_tasks, atol=atol)
        all_task_diffs.extend(task_diffs)
        if all(task_diff.passed for task_diff in task_diffs):
            passed_batches += 1
        if batch_index % 100 == 0:
            logger.info(f"[stage-compare] {stage_name}: {batch_index}/{total_batches} batches processed...")

    failed_batches = total_batches - passed_batches
    pass_rate = passed_batches / total_batches

    report = CompareReport(
        stage=stage_name,
        atol=atol,
        total_batches=total_batches,
        passed_batches=passed_batches,
        failed_batches=failed_batches,
        pass_rate=pass_rate,
        fields=_summarize_task_diffs(all_task_diffs),
        failures=[task_diff for task_diff in all_task_diffs if not task_diff.passed],
        profile_name=profile_name,
    )
    report.write_json(report_path)

    status = "PASSED" if pass_rate >= pass_threshold else "FAILED"
    logger.info(
        f"[stage-compare] {status}  {passed_batches}/{total_batches} ({pass_rate * 100:.1f}%)  report: {report_path}"
    )

    return StageCompareResult(report=report, report_path=report_path, pass_threshold=pass_threshold)
