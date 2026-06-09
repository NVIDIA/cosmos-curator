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
"""Stage 1: per-clip metadata + caption comparison.

The :class:`MetadataStage` actor holds one BGE caption model per Ray worker
and dispatches each batch through module-level comparators. Ray Data wiring
(pool sizing, runtime env) lives in :mod:`...driver`. Only
``artifact_kind == "clip"`` reaches the actor -- the driver filters non-clip
artifacts before submission.
"""

import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import pyarrow as pa
import smart_open  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.split_comparison.caption_embedding import (
    cosine_similarity_batch,
    load_caption_model,
)
from cosmos_curator.pipelines.video.split_comparison.config import (
    CaptionPolicy,
    ScoreTolerance,
    SplitComparisonConfig,
)
from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    Issue,
    make_issue,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

_FEATURE_STRUCTURE = "metadata_structure"
_FEATURE_AESTHETIC = "aesthetic_score"
_FEATURE_MOTION = "motion_score"
_FEATURE_CAPTIONS = "captions"

_MOTION_SCORE_NESTED_FIELDS: tuple[str, ...] = ("global_mean", "per_patch_min_256")


class CaptionJob(NamedTuple):
    """One caption-window pair queued for an embedding similarity check.

    Jobs from every clip in a Ray Data batch are embedded together in one
    cross-clip :func:`SentenceTransformer.encode` call.
    """

    clip_id: str
    video_key: str
    window: Mapping[str, Any]
    text_a: str
    text_b: str


class ScoreKind(Enum):
    """Classification of a raw JSON value read out of a per-clip metadata field.

    Used by :func:`_read_score`. Callers branch on the kind to decide whether
    to use the numeric value, treat the field as absent, or emit a
    ``metadata_value_invalid_type`` issue.
    """

    PRESENT = auto()  # finite numeric value, safe to compare
    MISSING = auto()  # key absent / null / NaN / +-inf -- legitimately not present
    CORRUPT = auto()  # producer wrote a non-numeric value (bool / str / list / ...) -- bug upstream


class MetadataRead(Enum):
    """Outcome of reading one output's per-clip metadata JSON.

    Distinguishes a legitimately-absent file from one that exists but can't be
    used, so the caller can emit a precise issue (``metadata_one_sided`` vs
    ``metadata_unreadable``) instead of reporting every failure as "missing".
    """

    OK = auto()  # parsed to a JSON object
    MISSING = auto()  # file absent (local FileNotFoundError / S3 NoSuchKey)
    UNREADABLE = auto()  # present but corrupt/truncated JSON, non-object payload, or transport error


class MetadataStage:
    """Compares per-clip metadata (aesthetic, motion, captions); owns one caption model per Ray worker."""

    # Pixi env this actor runs in. Read by ``run_metadata_stage`` to build the
    # Ray ``runtime_env``; matches the project convention of stages declaring
    # their env. Override per deployment if heavier caption deps move to a
    # dedicated env.
    conda_env_name: ClassVar[str] = "default"

    def __init__(
        self,
        output_a: str,
        output_b: str,
        profile_name: str,
        config: SplitComparisonConfig,
    ) -> None:
        """Bind run-wide args and eager-resolve per-output ``smart_open`` params + caption model.

        ``output_a`` / ``output_b`` are immutable for the actor's lifetime, so we
        resolve their params once here. Caption model loading is skipped when
        ``config.compare_captions`` is False to avoid the BGE weights lookup on
        metadata-only audits.
        """
        self._output_a = output_a
        self._output_b = output_b
        self._config = config
        self._caption_model = load_caption_model(config.caption.model_id) if config.compare_captions else None
        self._params_a: Mapping[str, Any] = storage_utils.get_smart_open_params(output_a, profile_name=profile_name)
        self._params_b: Mapping[str, Any] = storage_utils.get_smart_open_params(output_b, profile_name=profile_name)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Dispatch the whole batch through :func:`_process_batch`; return the issues table."""
        rows = _process_batch(
            batch.to_pylist(),
            output_a=self._output_a,
            output_b=self._output_b,
            params_a=self._params_a,
            params_b=self._params_b,
            config=self._config,
            caption_model=self._caption_model,
        )
        return pa.Table.from_pylist(rows, schema=ISSUE_SCHEMA)


def _process_batch(  # noqa: PLR0913
    rows: Sequence[Mapping[str, Any]],
    *,
    output_a: str,
    output_b: str,
    params_a: Mapping[str, Any],
    params_b: Mapping[str, Any],
    config: SplitComparisonConfig,
    caption_model: "SentenceTransformer | None",
) -> list[Issue]:
    """Two-phase per-batch dispatch.

    Phase 1: walk every row, run aesthetic/motion comparisons, collect caption
    jobs. Caption job collection is skipped when the actor has no caption model.

    Phase 2: if any caption jobs accumulated, embed them all in one
    ``encode()`` call and emit per-job below-threshold issues.
    """
    all_issues: list[Issue] = []
    all_jobs: list[CaptionJob] = []
    collect = caption_model is not None
    for row in rows:
        issues, jobs = _process_row_metadata_only(
            row,
            output_a=output_a,
            output_b=output_b,
            params_a=params_a,
            params_b=params_b,
            config=config,
            collect_caption_jobs=collect,
        )
        all_issues.extend(issues)
        all_jobs.extend(jobs)
    if all_jobs and caption_model is not None:
        all_issues.extend(_emit_caption_issues(all_jobs, caption_model, config.caption))
    return all_issues


def _process_row_metadata_only(  # noqa: PLR0913 -- routes a row through the actor-bound state
    row: Mapping[str, Any],
    *,
    output_a: str,
    output_b: str,
    params_a: Mapping[str, Any],
    params_b: Mapping[str, Any],
    config: SplitComparisonConfig,
    collect_caption_jobs: bool,
) -> tuple[list[Issue], list[CaptionJob]]:
    """Phase-1 work for one clip: cheap comparisons + (optional) caption job collection.

    Returns ``(issues, caption_jobs)``. Caption embedding is deferred to
    :func:`_emit_caption_issues` for cross-clip batching (see
    :func:`_process_batch`). ``collect_caption_jobs`` gates whether caption
    windows are walked at all; off when ``config.compare_captions`` is False.
    """
    clip_id = row["clip_id"]
    video_key = row["video_key"]
    in_a, in_b = row["in_a"], row["in_b"]

    if not (in_a and in_b):
        return (
            [
                make_issue(
                    code="metadata_one_sided",
                    message="Clip metadata present on only one output",
                    feature=_FEATURE_STRUCTURE,
                    video=video_key,
                    clip=clip_id,
                    output="b" if in_a else "a",
                ),
            ],
            [],
        )

    meta_a, status_a = _read_metadata(output_a, clip_id, params_a)
    meta_b, status_b = _read_metadata(output_b, clip_id, params_b)
    if meta_a is None or meta_b is None:
        return (_metadata_read_issues(clip_id, video_key, status_a=status_a, status_b=status_b), [])

    issues: list[Issue] = []
    issues.extend(compare_aesthetic_score(clip_id, video_key, meta_a, meta_b, config.aesthetic))
    issues.extend(compare_motion_score(clip_id, video_key, meta_a, meta_b, config.motion))
    jobs = _collect_caption_jobs(clip_id, video_key, meta_a, meta_b) if collect_caption_jobs else []
    return issues, jobs


def _metadata_read_issues(
    clip_id: str,
    video_key: str,
    *,
    status_a: MetadataRead,
    status_b: MetadataRead,
) -> list[Issue]:
    """Emit a per-output issue for each side whose metadata could not be loaded.

    The clip cleared the summary-parity check, so both files are expected to
    exist. A side that is :attr:`MetadataRead.MISSING` yields ``metadata_one_sided``;
    one that is :attr:`MetadataRead.UNREADABLE` yields ``metadata_unreadable``.
    Emitting per side avoids falsely blaming output A when both are absent.
    """
    issues: list[Issue] = []
    for output, status in (("a", status_a), ("b", status_b)):
        if status is MetadataRead.MISSING:
            issues.append(
                make_issue(
                    code="metadata_one_sided",
                    message=f"Clip metadata JSON missing on output {output.upper()} despite summary parity",
                    feature=_FEATURE_STRUCTURE,
                    video=video_key,
                    clip=clip_id,
                    output=output,
                ),
            )
        elif status is MetadataRead.UNREADABLE:
            issues.append(
                make_issue(
                    code="metadata_unreadable",
                    message=f"Clip metadata JSON on output {output.upper()} is present but could not be read",
                    feature=_FEATURE_STRUCTURE,
                    video=video_key,
                    clip=clip_id,
                    output=output,
                ),
            )
    return issues


def _read_metadata(
    output_root: str,
    clip_id: str,
    params: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, MetadataRead]:
    """Open the per-clip metadata JSON at ``<output_root>/metas/v0/<clip_id>.json``.

    Pure I/O routine -- holds no state. Caller resolves ``output_root`` /
    ``params`` from the actor. Returns ``(payload, status)``:

    * ``(dict, OK)`` -- parsed to a JSON object.
    * ``(None, MISSING)`` -- file absent (local ``FileNotFoundError`` or S3
      ``NoSuchKey`` wrapped in ``OSError``); silent, the expected one-sided case.
    * ``(None, UNREADABLE)`` -- present but unusable (corrupt/truncated JSON,
      transport error, or a payload that isn't a JSON object); logged.

    NOTE: transitional twin of ``measure_stage.read_clip_metadata`` (same I/O,
    this returns the ``MetadataRead`` enum, that returns presence bools). When
    CVC-1029 retires this module the two collapse into one shared reader.
    """
    path = storage_utils.get_full_path(output_root, "metas", "v0", f"{clip_id}.json")
    try:
        with smart_open.open(str(path), "rb", **params) as stream:
            payload = json.load(stream)
    except Exception as exc:  # noqa: BLE001 -- missing is silent, anything else is unreadable+logged
        if storage_utils.is_missing_object_error(exc):
            return None, MetadataRead.MISSING
        logger.exception("Failed to load metadata JSON: clip={} path={}", clip_id, path)
        return None, MetadataRead.UNREADABLE
    if not isinstance(payload, dict):
        logger.warning("Metadata JSON is not an object: clip={} path={} type={}", clip_id, path, type(payload).__name__)
        return None, MetadataRead.UNREADABLE
    return payload, MetadataRead.OK


def compare_aesthetic_score(
    clip_id: str,
    video_key: str,
    meta_a: Mapping[str, Any],
    meta_b: Mapping[str, Any],
    policy: ScoreTolerance,
) -> list[Issue]:
    """Compare scalar ``aesthetic_score`` under abs/rel tolerance, when present on both outputs.

    Corrupt values on either output (non-numeric types) produce a
    ``metadata_value_invalid_type`` issue per offending output and skip the
    tolerance comparison -- you can't compare against corrupt data. Asymmetric
    presence (one output has it, the other doesn't) produces a
    ``metadata_value_one_sided`` issue tagged with the missing output.
    """
    raw_a = meta_a.get("aesthetic_score")
    raw_b = meta_b.get("aesthetic_score")
    value_a, kind_a = _read_score(raw_a)
    value_b, kind_b = _read_score(raw_b)
    field = "aesthetic_score"
    corrupt_issues = _corrupt_value_issues(
        clip_id=clip_id,
        video_key=video_key,
        feature=_FEATURE_AESTHETIC,
        field=field,
        raw_a=raw_a,
        raw_b=raw_b,
        kind_a=kind_a,
        kind_b=kind_b,
    )
    if corrupt_issues:
        return corrupt_issues
    one_sided_issues = _one_sided_value_issue(
        clip_id=clip_id,
        video_key=video_key,
        feature=_FEATURE_AESTHETIC,
        field=field,
        kind_a=kind_a,
        kind_b=kind_b,
    )
    if one_sided_issues:
        return one_sided_issues
    if kind_a is not ScoreKind.PRESENT or kind_b is not ScoreKind.PRESENT:
        # Both MISSING -- legit "neither output wrote this score". Stay silent.
        return []
    if _within_tolerance(value_a, value_b, policy):
        return []
    return [
        make_issue(
            code="aesthetic_score_mismatch",
            message="Aesthetic score differs between outputs",
            feature=_FEATURE_AESTHETIC,
            video=video_key,
            clip=clip_id,
            field=field,
            details={"a": value_a, "b": value_b},
        ),
    ]


def compare_motion_score(
    clip_id: str,
    video_key: str,
    meta_a: Mapping[str, Any],
    meta_b: Mapping[str, Any],
    policy: ScoreTolerance,
) -> list[Issue]:
    """Compare nested ``motion_score`` fields (``global_mean``, ``per_patch_min_256``) under tolerance.

    Corrupt values on either output (non-numeric types) produce a
    ``metadata_value_invalid_type`` issue per offending output and skip the
    tolerance comparison for that nested field. Asymmetric presence
    (field on one output but not the other) produces a
    ``metadata_value_one_sided`` issue tagged with the missing output.
    """
    motion_a = meta_a.get("motion_score")
    motion_b = meta_b.get("motion_score")
    # Branch on the positive ``isinstance`` form so mypy narrows both variables to
    # ``Mapping`` inside the comparison block (no asserts, no casts).
    if isinstance(motion_a, Mapping) and isinstance(motion_b, Mapping):
        issues: list[Issue] = []
        for nested_field in _MOTION_SCORE_NESTED_FIELDS:
            raw_a = motion_a.get(nested_field)
            raw_b = motion_b.get(nested_field)
            value_a, kind_a = _read_score(raw_a)
            value_b, kind_b = _read_score(raw_b)
            field_name = f"motion_score.{nested_field}"
            corrupt_issues = _corrupt_value_issues(
                clip_id=clip_id,
                video_key=video_key,
                feature=_FEATURE_MOTION,
                field=field_name,
                raw_a=raw_a,
                raw_b=raw_b,
                kind_a=kind_a,
                kind_b=kind_b,
            )
            if corrupt_issues:
                issues.extend(corrupt_issues)
                continue
            one_sided_issues = _one_sided_value_issue(
                clip_id=clip_id,
                video_key=video_key,
                feature=_FEATURE_MOTION,
                field=field_name,
                kind_a=kind_a,
                kind_b=kind_b,
            )
            if one_sided_issues:
                issues.extend(one_sided_issues)
                continue
            if kind_a is not ScoreKind.PRESENT or kind_b is not ScoreKind.PRESENT:
                # Both MISSING -- legit "neither output wrote this nested field". Stay silent.
                continue
            if _within_tolerance(value_a, value_b, policy):
                continue
            issues.append(
                make_issue(
                    code="motion_score_mismatch",
                    message=f"motion_score.{nested_field} differs between outputs",
                    feature=_FEATURE_MOTION,
                    video=video_key,
                    clip=clip_id,
                    field=field_name,
                    details={"a": value_a, "b": value_b},
                ),
            )
        return issues
    if not isinstance(motion_a, Mapping) and not isinstance(motion_b, Mapping):
        # Both outputs lack a structured motion_score -- legitimate dual-absence.
        return []
    # Exactly one side has a Mapping; the other is missing or non-Mapping.
    missing_output = "b" if isinstance(motion_a, Mapping) else "a"
    return [
        make_issue(
            code="metadata_value_one_sided",
            message=f"motion_score present on only one output (missing on {missing_output.upper()})",
            feature=_FEATURE_MOTION,
            video=video_key,
            clip=clip_id,
            field="motion_score",
            output=missing_output,
        ),
    ]


def _corrupt_value_issues(  # noqa: PLR0913 -- comparator carries the full call context per check
    *,
    clip_id: str,
    video_key: str,
    feature: str,
    field: str,
    raw_a: Any,  # noqa: ANN401 -- arbitrary JSON value
    raw_b: Any,  # noqa: ANN401 -- arbitrary JSON value
    kind_a: ScoreKind,
    kind_b: ScoreKind,
) -> list[Issue]:
    """Emit one ``metadata_value_invalid_type`` issue per output whose kind is CORRUPT."""
    issues: list[Issue] = []
    for output, raw_value, kind in (("a", raw_a, kind_a), ("b", raw_b, kind_b)):
        if kind is ScoreKind.CORRUPT:
            issues.append(
                make_issue(
                    code="metadata_value_invalid_type",
                    message=f"{field} on output {output.upper()} is not a numeric value",
                    feature=feature,
                    video=video_key,
                    clip=clip_id,
                    field=field,
                    output=output,
                    details={
                        "raw_value": str(raw_value),
                        "raw_type": type(raw_value).__name__,
                    },
                ),
            )
    return issues


def _one_sided_value_issue(  # noqa: PLR0913 -- comparator carries the full call context per check
    *,
    clip_id: str,
    video_key: str,
    feature: str,
    field: str,
    kind_a: ScoreKind,
    kind_b: ScoreKind,
) -> list[Issue]:
    """Emit a ``metadata_value_one_sided`` issue when exactly one output is PRESENT.

    Both PRESENT or both not-PRESENT stay silent. Callers should run the
    upstream corrupt-check (:func:`_corrupt_value_issues`) first so this
    helper sees only PRESENT/MISSING combinations.
    """
    a_present = kind_a is ScoreKind.PRESENT
    b_present = kind_b is ScoreKind.PRESENT
    if a_present == b_present:
        return []
    missing_output = "b" if a_present else "a"
    return [
        make_issue(
            code="metadata_value_one_sided",
            message=f"{field} present on only one output (missing on {missing_output.upper()})",
            feature=feature,
            video=video_key,
            clip=clip_id,
            field=field,
            output=missing_output,
        ),
    ]


def _read_score(value: Any) -> tuple[float, ScoreKind]:  # noqa: ANN401 -- accepts arbitrary JSON values
    """Canonicalize a raw JSON value into ``(float, ScoreKind)`` for the score comparators.

    PRESENT for finite numeric values; MISSING for ``None``, NaN, +/-inf;
    CORRUPT for anything else. Bools are CORRUPT explicitly because
    ``isinstance(True, int)`` is True in Python -- without the guard, a stray
    ``true`` in the JSON would silently become ``1.0``.

    The float is sentinel ``0.0`` for non-PRESENT kinds; callers must check
    the kind before using the value.
    """
    if value is None:
        return 0.0, ScoreKind.MISSING
    if isinstance(value, bool):
        # Bools satisfy isinstance(x, int); classify explicitly as CORRUPT.
        return 0.0, ScoreKind.CORRUPT
    if isinstance(value, (int, float)):
        return (float(value), ScoreKind.PRESENT) if math.isfinite(value) else (0.0, ScoreKind.MISSING)
    return 0.0, ScoreKind.CORRUPT


def _within_tolerance(value_a: float, value_b: float, policy: ScoreTolerance) -> bool:
    """Pass if ``a`` and ``b`` agree under either policy tolerance; relative check skipped when both are zero."""
    diff = abs(value_a - value_b)
    if diff <= policy.abs_tolerance:
        return True
    larger = max(abs(value_a), abs(value_b))
    return larger > 0 and diff / larger <= policy.rel_tolerance


def compare_captions(  # noqa: PLR0913 -- comparator owns the full call context per clip
    clip_id: str,
    video_key: str,
    meta_a: Mapping[str, Any],
    meta_b: Mapping[str, Any],
    model: "SentenceTransformer",
    policy: CaptionPolicy,
) -> list[Issue]:
    """Single-clip caption comparison; defers to :func:`_emit_caption_issues`.

    The production path runs :func:`_process_batch` for cross-clip batching;
    this entry point exists for direct callers / unit tests with one clip.
    """
    jobs = _collect_caption_jobs(clip_id, video_key, meta_a, meta_b)
    return _emit_caption_issues(jobs, model, policy) if jobs else []


def _collect_caption_jobs(
    clip_id: str,
    video_key: str,
    meta_a: Mapping[str, Any],
    meta_b: Mapping[str, Any],
) -> list[CaptionJob]:
    """Walk one clip's caption windows; return one :class:`CaptionJob` per divergent pair.

    Identical-text windows short-circuit (no embedding needed). One-sided
    windows still produce a job with an empty string on the missing output --
    matches the current ``compare_captions`` behavior where a half-empty pair
    falls below threshold and surfaces as a caption_similarity_below_threshold
    issue.
    """
    windows_a = _caption_windows(meta_a)
    windows_b = _caption_windows(meta_b)
    if not (windows_a or windows_b):
        return []
    jobs: list[CaptionJob] = []
    for key in sorted(set(windows_a) | set(windows_b)):
        w_a = windows_a.get(key)
        w_b = windows_b.get(key)
        text_a = _caption_text(w_a)
        text_b = _caption_text(w_b)
        if w_a is not None and w_b is not None and text_a == text_b:
            continue
        start, end = key
        window = {"start_frame": start, "end_frame": end}
        jobs.append(
            CaptionJob(
                clip_id=clip_id,
                video_key=video_key,
                window=window,
                text_a=text_a,
                text_b=text_b,
            ),
        )
    return jobs


def _caption_windows(metadata: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    windows = metadata.get("windows")
    if not isinstance(windows, list):
        return {}
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for window in windows:
        if not isinstance(window, Mapping):
            continue
        start = window.get("start_frame")
        end = window.get("end_frame")
        if isinstance(start, int) and isinstance(end, int):
            result[(start, end)] = window
    return result


def _caption_text(window: Mapping[str, Any] | None) -> str:
    if window is None:
        return ""
    value = window.get("qwen_caption")
    return value if isinstance(value, str) else ""


def _emit_caption_issues(
    jobs: Sequence[CaptionJob],
    model: "SentenceTransformer",
    policy: CaptionPolicy,
) -> list[Issue]:
    """Embed every job's text pair in one ``encode()`` call; emit below-threshold issues.

    One Python-level :func:`encode` covers all jobs in the batch;
    ``policy.encode_batch_size`` sets the internal chunk size.
    """
    if not jobs:
        return []
    texts_a = [job.text_a for job in jobs]
    texts_b = [job.text_b for job in jobs]
    similarities = cosine_similarity_batch(model, texts_a, texts_b, batch_size=policy.encode_batch_size)
    return [
        make_issue(
            code="caption_similarity_below_threshold",
            message=f"Caption similarity {float(sim):.3f} below threshold {policy.min_similarity:.3f}",
            feature=_FEATURE_CAPTIONS,
            video=job.video_key,
            clip=job.clip_id,
            field="caption",
            details={
                "start_frame": job.window["start_frame"],
                "end_frame": job.window["end_frame"],
                "similarity": float(sim),
                "threshold": policy.min_similarity,
                "a": job.text_a,
                "b": job.text_b,
            },
        )
        for job, sim in zip(jobs, similarities, strict=True)
        if float(sim) < policy.min_similarity
    ]
