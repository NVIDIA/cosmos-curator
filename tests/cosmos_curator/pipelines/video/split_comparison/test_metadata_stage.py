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
"""Tests for Stage 1 metadata comparison helpers and the MetadataStage actor."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.pipelines.video.split_comparison.clip_discovery import CLIP_ROW_SCHEMA
from cosmos_curator.pipelines.video.split_comparison.config import (
    CaptionPolicy,
    ScoreTolerance,
    SplitComparisonConfig,
)
from cosmos_curator.pipelines.video.split_comparison.driver import run_metadata_stage
from cosmos_curator.pipelines.video.split_comparison.metadata_stage import (
    CaptionJob,
    MetadataStage,
    ScoreKind,
    _collect_caption_jobs,
    _emit_caption_issues,
    _one_sided_value_issue,
    _process_batch,
    _read_score,
    _within_tolerance,
    compare_aesthetic_score,
    compare_captions,
    compare_motion_score,
)
from cosmos_curator.pipelines.video.split_comparison.result_model import ISSUE_SCHEMA


class _FakeCaptionModel:
    """Deterministic sentence-encoder for tests: hashes each string to a unit vector."""

    def __init__(self, *, identical_score: float = 1.0, default_score: float = 0.3) -> None:
        self._identical_score = identical_score
        self._default_score = default_score

    def encode(
        self,
        texts: list[str],
        **_kwargs: Any,  # noqa: ANN401 -- mirror sentence-transformers' open kwargs
    ) -> "np.ndarray[Any, Any]":
        """Return embeddings whose dot product is identical_score for equal text, default_score otherwise."""
        # First half is "output A", second half is "output B" by convention of the caller.
        half = len(texts) // 2
        embeddings = np.zeros((len(texts), 32), dtype=np.float32)
        for idx, text in enumerate(texts):
            embeddings[idx] = self._embed(text)
        # Adjust output-B embeddings so paired dot products match the fake's intent.
        for i in range(half):
            text_a = texts[i]
            text_b = texts[i + half]
            target = self._identical_score if text_a == text_b else self._default_score
            # Set embs_b[i] = target * embs_a[i] + sqrt(1 - target^2) * orthogonal
            base = embeddings[i]
            orth = self._orthogonal(base)
            embeddings[i + half] = target * base + (1 - target**2) ** 0.5 * orth
        return embeddings

    @staticmethod
    def _embed(text: str) -> "np.ndarray[Any, Any]":
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float32)
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec)
        return np.asarray(vec / norm) if norm else np.asarray(vec)

    @staticmethod
    def _orthogonal(vec: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
        # Build a vector orthogonal to `vec` of the same shape: any rotation suffices.
        orth = np.roll(vec, 1).copy()
        orth -= vec * float(orth @ vec)
        norm = np.linalg.norm(orth)
        return np.asarray(orth / norm) if norm else np.asarray(orth)


def _details(issue: dict[str, Any]) -> dict[str, Any]:
    return json.loads(issue["details"]) if issue.get("details") else {}


class _FakeStream:
    """Module-level fake of smart_open's binary-mode stream; tests substitute via monkeypatch."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._data


# --- _read_score / ScoreKind ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # PRESENT for finite numerics (int + float, positive / negative / zero).
        (5, (5.0, ScoreKind.PRESENT)),
        (-3, (-3.0, ScoreKind.PRESENT)),
        (0, (0.0, ScoreKind.PRESENT)),
        (3.14, (3.14, ScoreKind.PRESENT)),
        (-0.5, (-0.5, ScoreKind.PRESENT)),
        (0.0, (0.0, ScoreKind.PRESENT)),
        # MISSING for None and non-finite floats (NaN / +-inf are "legitimately absent").
        (None, (0.0, ScoreKind.MISSING)),
        (float("nan"), (0.0, ScoreKind.MISSING)),
        (float("inf"), (0.0, ScoreKind.MISSING)),
        (float("-inf"), (0.0, ScoreKind.MISSING)),
        # CORRUPT for bools: ``isinstance(True, int)`` is True in Python; the bool guard
        # is what keeps a stray JSON ``true`` from silently becoming 1.0.
        (True, (0.0, ScoreKind.CORRUPT)),
        (False, (0.0, ScoreKind.CORRUPT)),
        # CORRUPT for non-numeric types -- indicates an upstream metadata-writer bug.
        ("0.5", (0.0, ScoreKind.CORRUPT)),
        ([], (0.0, ScoreKind.CORRUPT)),
        ({}, (0.0, ScoreKind.CORRUPT)),
    ],
)
def test_read_score_classifies_into_kind(value: object, expected: tuple[float, ScoreKind]) -> None:
    """One row per (input shape -> kind) the canonicalizer is expected to produce."""
    assert _read_score(value) == expected


# --- _within_tolerance --------------------------------------------------------------


@pytest.mark.parametrize(
    ("value_a", "value_b", "policy", "within"),
    [
        # Exact match: any tolerance accepts (covers both nonzero and both-zero).
        (0.5, 0.5, ScoreTolerance(abs_tolerance=0.0, rel_tolerance=0.0), True),
        (0.0, 0.0, ScoreTolerance(abs_tolerance=0.0, rel_tolerance=0.0), True),
        # Diff within abs_tolerance short-circuits (rel never consulted).
        (0.500, 0.5009, ScoreTolerance(abs_tolerance=0.001, rel_tolerance=0.0), True),
        # Diff outside abs but within rel: 1.0 / 100 = 0.01 satisfies rel_tolerance=0.01.
        (99.0, 100.0, ScoreTolerance(abs_tolerance=0.0, rel_tolerance=0.01), True),
        # Both-zero short-circuits via abs check; rel has no defined ratio so it doesn't matter.
        (0.0, 0.0, ScoreTolerance(abs_tolerance=0.0, rel_tolerance=1.0), True),
        # Negative inputs go through abs(...) so the check is sign-agnostic.
        (-0.50, -0.5009, ScoreTolerance(abs_tolerance=0.001, rel_tolerance=0.0), True),
        # Drift exceeding both tolerances -- positive and negative.
        (0.5, 0.65, ScoreTolerance(abs_tolerance=1e-3, rel_tolerance=1e-3), False),
        (-0.5, -0.65, ScoreTolerance(abs_tolerance=1e-3, rel_tolerance=1e-3), False),
    ],
)
def test_within_tolerance(
    value_a: float,
    value_b: float,
    policy: ScoreTolerance,
    within: bool,  # noqa: FBT001 -- parametrize delivers this by name
) -> None:
    """One row per scenario: exact match, abs short-circuit, rel pass, drift, negatives."""
    assert _within_tolerance(value_a, value_b, policy) is within


# --- _one_sided_value_issue ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kind_a", "kind_b"),
    [
        # Both PRESENT: helper defers to the comparator's tolerance check.
        (ScoreKind.PRESENT, ScoreKind.PRESENT),
        # Both MISSING: legitimate "neither output wrote this field".
        (ScoreKind.MISSING, ScoreKind.MISSING),
        # Both CORRUPT / CORRUPT + MISSING: silent -- the corrupt-check upstream owns those issues.
        (ScoreKind.CORRUPT, ScoreKind.CORRUPT),
        (ScoreKind.CORRUPT, ScoreKind.MISSING),
    ],
)
def test_one_sided_silent_when_not_exactly_one_present(kind_a: ScoreKind, kind_b: ScoreKind) -> None:
    """Emits nothing whenever the count of PRESENT outputs isn't exactly one."""
    assert (
        _one_sided_value_issue(
            clip_id="clip-a",
            video_key="video.mp4",
            feature="aesthetic_score",
            field="aesthetic_score",
            kind_a=kind_a,
            kind_b=kind_b,
        )
        == []
    )


@pytest.mark.parametrize(
    ("kind_a", "kind_b", "missing_output", "feature", "field"),
    [
        # A PRESENT, B MISSING -> tag B; also pins that feature/field propagate.
        (ScoreKind.PRESENT, ScoreKind.MISSING, "b", "aesthetic_score", "aesthetic_score"),
        # A MISSING, B PRESENT -> tag A; uses motion to pin nested field propagation.
        (ScoreKind.MISSING, ScoreKind.PRESENT, "a", "motion_score", "motion_score.global_mean"),
    ],
)
def test_one_sided_emits_issue_tagged_with_missing_output(
    kind_a: ScoreKind,
    kind_b: ScoreKind,
    missing_output: str,
    feature: str,
    field: str,
) -> None:
    """Exactly one PRESENT -> one issue tagged with the missing output's letter."""
    issues = _one_sided_value_issue(
        clip_id="clip-a",
        video_key="video.mp4",
        feature=feature,
        field=field,
        kind_a=kind_a,
        kind_b=kind_b,
    )
    assert len(issues) == 1
    issue = issues[0]
    assert issue["code"] == "metadata_value_one_sided"
    assert issue["output"] == missing_output
    assert issue["field"] == field
    assert issue["feature"] == feature
    assert issue["clip"] == "clip-a"
    assert issue["video"] == "video.mp4"
    assert f"missing on {missing_output.upper()}" in issue["message"]


# --- compare_aesthetic_score ---------------------------------------------------------


def test_aesthetic_no_issue_when_both_missing() -> None:
    """Feature is treated as not-applicable when neither output reports it."""
    issues = compare_aesthetic_score("clip-a", "video.mp4", {}, {}, ScoreTolerance())
    assert issues == []


def test_aesthetic_flags_difference_outside_tolerance() -> None:
    """A real numeric difference exceeding abs_tolerance produces an issue with both values."""
    issues = compare_aesthetic_score(
        "clip-a",
        "video.mp4",
        {"aesthetic_score": 0.50},
        {"aesthetic_score": 0.65},
        ScoreTolerance(abs_tolerance=1e-3, rel_tolerance=1e-3),
    )
    assert len(issues) == 1
    issue = issues[0]
    assert issue["code"] == "aesthetic_score_mismatch"
    assert issue["clip"] == "clip-a"
    assert _details(issue) == {"a": 0.5, "b": 0.65}


# One-sided + corrupt-value wiring is covered by the ``_one_sided_value_issue`` and
# ``_read_score`` helper tests above; compare_aesthetic_score adds no behavior beyond
# routing through those, so we don't re-verify the symmetric A/B and corrupt-type
# permutations here. The motion comparator's consolidated test below pins the same
# wiring once for the nested-field iteration path.


def test_aesthetic_respects_abs_tolerance() -> None:
    """Differences within abs_tolerance are silent."""
    issues = compare_aesthetic_score(
        "clip-a",
        "video.mp4",
        {"aesthetic_score": 0.5},
        {"aesthetic_score": 0.5001},
        ScoreTolerance(abs_tolerance=1e-3),
    )
    assert issues == []


# --- compare_motion_score ------------------------------------------------------------


def test_motion_score_iterates_nested_fields_with_per_field_wiring() -> None:
    """For each nested motion_score field, the comparator wires through to the score helpers.

    One test, mixed paths: ``global_mean`` mismatches (mismatch wiring),
    ``per_patch_min_256`` is one-sided on B (one-sided wiring). Same helpers as
    aesthetic; this test specifically pins the nested-field iteration path, which
    the aesthetic comparator doesn't exercise.
    """
    issues = compare_motion_score(
        "clip-a",
        "video.mp4",
        {"motion_score": {"global_mean": 1.0}},  # global_mean differs; per_patch_min_256 absent
        {"motion_score": {"global_mean": 0.7, "per_patch_min_256": 0.4}},
        ScoreTolerance(abs_tolerance=1e-3),
    )
    codes_by_field = sorted((issue["field"], issue["code"]) for issue in issues)
    assert codes_by_field == [
        ("motion_score.global_mean", "motion_score_mismatch"),
        ("motion_score.per_patch_min_256", "metadata_value_one_sided"),
    ]


# --- compare_captions ----------------------------------------------------------------


def _meta_with_caption_windows(*windows: tuple[int, int, str]) -> dict[str, Any]:
    return {
        "windows": [{"start_frame": start, "end_frame": end, "qwen_caption": text} for start, end, text in windows],
    }


def test_captions_no_issue_for_identical_caption_strings() -> None:
    """Exact-text matches skip the model entirely (fast path)."""
    meta_a = _meta_with_caption_windows((0, 30, "a dog"), (30, 60, "a cat"))
    meta_b = _meta_with_caption_windows((0, 30, "a dog"), (30, 60, "a cat"))

    # Even with a model that would otherwise compute low similarity, identical strings short-circuit.
    issues = compare_captions(
        "clip-a",
        "video.mp4",
        meta_a,
        meta_b,
        _FakeCaptionModel(default_score=0.0),
        CaptionPolicy(min_similarity=0.9),
    )
    assert issues == []


def test_captions_emits_one_issue_per_below_threshold_window() -> None:
    """Each window pair whose cosine similarity is below threshold becomes its own row."""
    meta_a = _meta_with_caption_windows((0, 30, "a dog"), (30, 60, "a cat"))
    meta_b = _meta_with_caption_windows((0, 30, "a different dog"), (30, 60, "a cat"))

    issues = compare_captions(
        "clip-a",
        "video.mp4",
        meta_a,
        meta_b,
        _FakeCaptionModel(default_score=0.2),
        CaptionPolicy(min_similarity=0.85),
    )

    assert len(issues) == 1
    details = _details(issues[0])
    assert details["start_frame"] == 0
    assert details["end_frame"] == 30
    assert details["threshold"] == 0.85
    assert details["similarity"] == pytest.approx(0.2, abs=1e-5)
    assert details["a"] == "a dog"
    assert details["b"] == "a different dog"


def test_captions_skips_when_no_windows_on_either_side() -> None:
    """No caption windows present anywhere -> nothing to compare."""
    issues = compare_captions(
        "clip-a",
        "video.mp4",
        {},
        {},
        _FakeCaptionModel(),
        CaptionPolicy(),
    )
    assert issues == []


# --- MetadataStage actor + run_metadata_stage ---------------------------------------


def _stub_storage(monkeypatch: pytest.MonkeyPatch, payload_by_path: Mapping[str, dict[str, Any]]) -> None:
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.metadata_stage.storage_utils.get_full_path",
        Path,
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.metadata_stage.storage_utils.path_exists",
        lambda path, _client: str(path) in payload_by_path,
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.metadata_stage.storage_utils.get_storage_client",
        lambda _root, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.metadata_stage.storage_utils.get_smart_open_params",
        lambda _root, **_kwargs: {},
    )

    def fake_open(
        path: str,
        mode: str,  # noqa: ARG001
        **_kwargs: object,
    ) -> _FakeStream:
        if path not in payload_by_path:
            raise FileNotFoundError(path)
        return _FakeStream(json.dumps(payload_by_path[path]).encode("utf-8"))

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.metadata_stage.smart_open.open",
        fake_open,
    )


def _row(
    *, clip_id: str = "clip-a", in_a: bool = True, in_b: bool = True, artifact_kind: str = "clip"
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "video_key": "video.mp4",
        "artifact_kind": artifact_kind,
        "in_a": in_a,
        "in_b": in_b,
    }


def _meta_path(output_root: str, clip_id: str) -> str:
    return str(Path(output_root) / "metas" / "v0" / f"{clip_id}.json")


def _drive(
    row: dict[str, Any],
    *,
    config: SplitComparisonConfig | None = None,
    caption_model: object | None = None,
    output_a: str = "/a",
    output_b: str = "/b",
) -> list[dict[str, Any]]:
    """Test helper: drive ``_process_batch`` with a single-row batch (storage stubs are module-level)."""
    return _process_batch(  # type: ignore[arg-type, return-value]
        [row],
        output_a=output_a,
        output_b=output_b,
        params_a={},
        params_b={},
        config=config or SplitComparisonConfig(output_a=output_a, output_b=output_b),
        caption_model=caption_model,
    )


def test_metadata_stage_emits_metadata_one_sided_for_asymmetric_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clip listed as in_a=True / in_b=False reports a metadata_one_sided issue."""
    _stub_storage(monkeypatch, {})

    issues = _drive(_row(in_b=False))

    assert [issue["code"] for issue in issues] == ["metadata_one_sided"]
    assert issues[0]["output"] == "b"


def test_metadata_stage_runs_all_three_compares_when_both_sides_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aesthetic + motion + captions all run when both metadata payloads are loadable."""
    meta_a = {
        "aesthetic_score": 0.5,
        "motion_score": {"global_mean": 1.0, "per_patch_min_256": 0.2},
        "windows": [{"start_frame": 0, "end_frame": 30, "qwen_caption": "a dog"}],
    }
    meta_b = {
        "aesthetic_score": 0.65,  # outside tolerance
        "motion_score": {"global_mean": 0.8, "per_patch_min_256": 0.2},  # global_mean off
        "windows": [{"start_frame": 0, "end_frame": 30, "qwen_caption": "a cat"}],
    }
    _stub_storage(
        monkeypatch,
        {_meta_path("/a", "clip-a"): meta_a, _meta_path("/b", "clip-a"): meta_b},
    )

    issues = _drive(
        _row(),
        config=SplitComparisonConfig(
            output_a="/a",
            output_b="/b",
            aesthetic=ScoreTolerance(abs_tolerance=1e-3),
            motion=ScoreTolerance(abs_tolerance=1e-3),
            caption=CaptionPolicy(min_similarity=0.85),
        ),
        caption_model=_FakeCaptionModel(),
    )
    codes = sorted(issue["code"] for issue in issues)

    assert codes == [
        "aesthetic_score_mismatch",
        "caption_similarity_below_threshold",
        "motion_score_mismatch",
    ]


def test_metadata_stage_skips_caption_work_when_compare_captions_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """compare_captions=False: the actor's emitted issues never include caption similarity findings."""
    meta_a = {
        "aesthetic_score": 0.5,
        "motion_score": {"global_mean": 1.0, "per_patch_min_256": 0.2},
        "windows": [{"start_frame": 0, "end_frame": 30, "qwen_caption": "a dog"}],
    }
    meta_b = {
        "aesthetic_score": 0.65,  # outside tolerance -> aesthetic issue still fires
        "motion_score": {"global_mean": 1.0, "per_patch_min_256": 0.2},
        "windows": [{"start_frame": 0, "end_frame": 30, "qwen_caption": "a cat"}],
    }
    _stub_storage(
        monkeypatch,
        {_meta_path("/a", "clip-a"): meta_a, _meta_path("/b", "clip-a"): meta_b},
    )

    stage = MetadataStage(
        output_a="/a",
        output_b="/b",
        profile_name="default",
        config=SplitComparisonConfig(
            output_a="/a",
            output_b="/b",
            aesthetic=ScoreTolerance(abs_tolerance=1e-3),
            compare_captions=False,
        ),
    )
    out = stage(pa.Table.from_pylist([_row()], schema=CLIP_ROW_SCHEMA))

    codes = sorted(out["code"].to_pylist())
    assert "caption_similarity_below_threshold" not in codes
    assert "aesthetic_score_mismatch" in codes  # non-caption comparisons still run


def test_metadata_stage_emits_metadata_one_sided_when_payload_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-clip metadata JSON missing on one output surfaces as metadata_one_sided."""
    meta_a = {"aesthetic_score": 0.5}
    _stub_storage(monkeypatch, {_meta_path("/a", "clip-a"): meta_a})  # /b path absent

    issues = _drive(_row())

    assert [issue["code"] for issue in issues] == ["metadata_one_sided"]
    assert issues[0]["output"] == "b"


def test_metadata_stage_emits_one_sided_per_side_when_both_payloads_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both per-clip metadata JSONs absent -> one metadata_one_sided row per output, not a single A-blaming row."""
    _stub_storage(monkeypatch, {})  # neither /a nor /b path present

    issues = _drive(_row())

    assert [issue["code"] for issue in issues] == ["metadata_one_sided", "metadata_one_sided"]
    assert sorted(issue["output"] for issue in issues) == ["a", "b"]


def test_metadata_stage_emits_metadata_unreadable_for_corrupt_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncated/corrupt metadata JSON surfaces as metadata_unreadable, not metadata_one_sided ('missing')."""
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.metadata_stage.storage_utils.get_full_path",
        Path,
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.metadata_stage.storage_utils.get_smart_open_params",
        lambda _root, **_kwargs: {},
    )
    good = json.dumps({"aesthetic_score": 0.5}).encode("utf-8")
    truncated = b'{"aesthetic_score": 0.5'  # valid prefix, never closed -> JSONDecodeError

    def fake_open(path: str, mode: str, **_kwargs: object) -> _FakeStream:  # noqa: ARG001
        return _FakeStream(good if str(path) == _meta_path("/b", "clip-a") else truncated)

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.metadata_stage.smart_open.open",
        fake_open,
    )

    issues = _drive(_row())

    assert [issue["code"] for issue in issues] == ["metadata_unreadable"]
    assert issues[0]["output"] == "a"


def test_collect_caption_jobs_skips_identical_windows() -> None:
    """Identical-text windows short-circuit (no embedding needed); divergent ones become jobs."""
    meta_a = _meta_with_caption_windows((0, 30, "a dog"), (30, 60, "a cat"))
    meta_b = _meta_with_caption_windows((0, 30, "a dog"), (30, 60, "two cats"))

    jobs = _collect_caption_jobs("clip-x", "video.mp4", meta_a, meta_b)

    assert len(jobs) == 1
    assert jobs[0].clip_id == "clip-x"
    assert jobs[0].video_key == "video.mp4"
    assert jobs[0].window == {"start_frame": 30, "end_frame": 60}
    assert jobs[0].text_a == "a cat"
    assert jobs[0].text_b == "two cats"


def test_emit_caption_issues_batches_across_clips() -> None:
    """_emit_caption_issues runs ONE model.encode call for jobs spanning multiple clips."""
    encode_calls = {"hit": 0}

    class _CountingModel:
        def encode(
            self,
            texts: list[str],
            **_kwargs: Any,  # noqa: ANN401 -- mirrors sentence-transformers' open kwargs
        ) -> "np.ndarray[Any, Any]":
            encode_calls["hit"] += 1
            # Force all pairs below threshold by returning unit vectors with paired dot ~ 0.
            n = len(texts) // 2
            embs = np.zeros((len(texts), 32), dtype=np.float32)
            for i in range(n):
                embs[i, 0] = 1.0
                embs[i + n, 1] = 1.0
            return embs

    jobs = [
        CaptionJob(
            clip_id="clip-1",
            video_key="video-a.mp4",
            window={"start_frame": 0, "end_frame": 30},
            text_a="x",
            text_b="y",
        ),
        CaptionJob(
            clip_id="clip-2",
            video_key="video-b.mp4",
            window={"start_frame": 30, "end_frame": 60},
            text_a="p",
            text_b="q",
        ),
    ]

    issues = _emit_caption_issues(jobs, _CountingModel(), CaptionPolicy(min_similarity=0.5))  # type: ignore[arg-type]

    assert encode_calls["hit"] == 1
    codes = sorted((issue["clip"], issue["video"]) for issue in issues)
    assert codes == [("clip-1", "video-a.mp4"), ("clip-2", "video-b.mp4")]


def test_process_batch_emits_caption_issues_for_each_diverging_clip_in_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-clip batch surfaces a caption issue for every clip whose captions diverge."""
    meta_a_clip1 = {
        "aesthetic_score": 0.5,
        "windows": [{"start_frame": 0, "end_frame": 30, "qwen_caption": "a dog"}],
    }
    meta_b_clip1 = {
        "aesthetic_score": 0.5,
        "windows": [{"start_frame": 0, "end_frame": 30, "qwen_caption": "a cat"}],
    }
    meta_a_clip2 = {
        "aesthetic_score": 0.5,
        "windows": [{"start_frame": 0, "end_frame": 30, "qwen_caption": "a house"}],
    }
    meta_b_clip2 = {
        "aesthetic_score": 0.5,
        "windows": [{"start_frame": 0, "end_frame": 30, "qwen_caption": "a car"}],
    }
    _stub_storage(
        monkeypatch,
        {
            _meta_path("/a", "clip-1"): meta_a_clip1,
            _meta_path("/b", "clip-1"): meta_b_clip1,
            _meta_path("/a", "clip-2"): meta_a_clip2,
            _meta_path("/b", "clip-2"): meta_b_clip2,
        },
    )

    issues = _process_batch(
        [_row(clip_id="clip-1"), _row(clip_id="clip-2")],
        output_a="/a",
        output_b="/b",
        params_a={},
        params_b={},
        config=SplitComparisonConfig(output_a="/a", output_b="/b", caption=CaptionPolicy(min_similarity=0.85)),
        caption_model=_FakeCaptionModel(default_score=0.2),
    )

    caption_issues = [issue for issue in issues if issue["code"] == "caption_similarity_below_threshold"]
    clips_with_caption_issues = sorted({issue["clip"] for issue in caption_issues})
    assert clips_with_caption_issues == ["clip-1", "clip-2"]


def test_run_metadata_stage_returns_empty_when_no_passed_clips() -> None:
    """Filtered-only clip tables short-circuit since Stage 1 doesn't compare filtered clip metadata."""
    only_filtered = pa.Table.from_pylist(
        [
            {
                "clip_id": "clip-a",
                "video_key": "video.mp4",
                "in_a": True,
                "in_b": True,
                "artifact_kind": "filtered_clip",
            },
        ],
        schema=CLIP_ROW_SCHEMA,
    )

    out = run_metadata_stage(only_filtered, config=SplitComparisonConfig(output_a="/a", output_b="/b"))

    assert out.schema == ISSUE_SCHEMA
    assert out.num_rows == 0
