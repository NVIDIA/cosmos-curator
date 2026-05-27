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
"""Tests for video-level feature comparison."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_curator.pipelines.video.output_comparison.caption_comparator import CaptionFeatureComparator
from cosmos_curator.pipelines.video.output_comparison.compare_features import (
    _canonical_load_worker_constructor_kwargs,
    compare_features,
)
from cosmos_curator.pipelines.video.output_comparison.feature_plan import (
    ClipFeaturePlan,
    FeatureComparisonContext,
    FeatureComparisonResult,
    ResolvedFeaturePlan,
)
from cosmos_curator.pipelines.video.output_comparison.report import FeatureComparison
from cosmos_curator.pipelines.video.output_comparison.score_comparator import (
    AESTHETIC_SCORE_FEATURE_NAME,
    MOTION_SCORE_FEATURE_NAME,
    ScoreFeatureComparator,
)
from cosmos_curator.pipelines.video.output_comparison.summary_schema import OutputSummary
from cosmos_curator.pipelines.video.output_comparison.video_planning import (
    build_clip_comparison_specs,
    build_video_comparison_specs,
)
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec, VideoComparisonSpec

from .conftest import summary, video_summary


class _FakeResolvedComparator:
    @property
    def name(self) -> str:
        return "fake_resolved"

    def build_plan(self, context: FeatureComparisonContext) -> ResolvedFeaturePlan:
        return ResolvedFeaturePlan(
            feature_name=self.name,
            result=FeatureComparisonResult(
                issues=(),
                comparison=FeatureComparison(status="passed", metrics={"planned_videos": len(context.specs)}),
            ),
        )


class _FakeMappedResolvedComparator:
    @property
    def name(self) -> str:
        return "fake_mapped"

    def build_plan(self, context: FeatureComparisonContext) -> ResolvedFeaturePlan:
        return ResolvedFeaturePlan(
            feature_name=self.name,
            result={
                "fake_resolved": FeatureComparisonResult(
                    issues=(),
                    comparison=FeatureComparison(status="passed", metrics={"planned_videos": len(context.specs)}),
                )
            },
        )


class _FakeClipLoadWorker:
    def __init__(self, marker: str) -> None:
        self._marker = marker

    def __call__(self, row: dict[str, object]) -> dict[str, object]:
        return row | {"loaded_marker": self._marker}


class _FakeClipComparator:
    @property
    def name(self) -> str:
        return "fake_clip"

    def build_plan(self, context: FeatureComparisonContext) -> ClipFeaturePlan:
        return ClipFeaturePlan(
            feature_name=self.name,
            clip_specs=build_clip_comparison_specs(context.specs),
            load_worker_class=_FakeClipLoadWorker,
            load_worker_constructor_kwargs={"marker": "loaded"},
            compare_row=self._compare_row,
            reduce_rows=self._reduce_rows,
        )

    def _compare_row(self, row: dict[str, object]) -> dict[str, object]:
        return {
            "video_key": row["video_key"],
            "clip_id": row["clip_id"],
            "marker": row["loaded_marker"],
        }

    def _reduce_rows(self, rows: Sequence[dict[str, object]]) -> FeatureComparisonResult:
        return FeatureComparisonResult(
            issues=(),
            comparison=FeatureComparison(status="passed", metrics={"rows": len(rows)}),
        )


class _FakeNonSerializableClipComparator:
    def __init__(self, *, load_group_id: str | None = None) -> None:
        self._load_group_id = load_group_id

    @property
    def name(self) -> str:
        return "fake_nonserializable_clip"

    def build_plan(self, context: FeatureComparisonContext) -> ClipFeaturePlan:
        _ = context
        return ClipFeaturePlan(
            feature_name=self.name,
            clip_specs=(),
            load_worker_class=_FakeClipLoadWorker,
            load_worker_constructor_kwargs={"marker": object()},
            compare_row=self._compare_row,
            reduce_rows=self._reduce_rows,
            load_group_id=self._load_group_id,
        )

    def _compare_row(self, row: dict[str, object]) -> dict[str, object]:
        return row

    def _reduce_rows(self, rows: Sequence[dict[str, object]]) -> FeatureComparisonResult:
        return FeatureComparisonResult(
            issues=(),
            comparison=FeatureComparison(status="passed", metrics={"rows": len(rows)}),
        )


def _output_summary() -> OutputSummary:
    return OutputSummary.from_json_dict(
        summary(
            **{
                "num_processed_videos": 2,
                "no-caption.mp4": video_summary(clips=["clip-b"]),
                "video.mp4": video_summary(clips=["clip-a"])
                | {
                    "num_clips_with_caption": 1,
                    "num_caption_windows": 1,
                },
                "total_num_clips_with_caption": 1,
                "total_num_caption_windows": 1,
            }
        )
    )


def test_build_video_comparison_specs_uses_summary_video_units(tmp_path: Path) -> None:
    """Video compare specs are derived from per-video summary clip UUID lists."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"

    specs = build_video_comparison_specs(output_a, output_b, _output_summary(), _output_summary())

    assert specs == (
        VideoComparisonSpec(
            video_key="no-caption.mp4",
            output_a=str(output_a),
            output_b=str(output_b),
            clips_a=("clip-b",),
            clips_b=("clip-b",),
        ),
        VideoComparisonSpec(
            video_key="video.mp4",
            output_a=str(output_a),
            output_b=str(output_b),
            clips_a=("clip-a",),
            clips_b=("clip-a",),
        ),
    )


def test_build_video_comparison_specs_limits_to_output_a_video_order(tmp_path: Path) -> None:
    """Limited video compare specs use the first N video keys from output A."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    summary_a_data = summary()
    del summary_a_data["video.mp4"]
    summary_a_data["first.mp4"] = video_summary(clips=["clip-a-1"])
    summary_a_data["second.mp4"] = video_summary(clips=["clip-a-2"])
    summary_a_data["third.mp4"] = video_summary(clips=["clip-a-3"])
    summary_b_data = summary()
    del summary_b_data["video.mp4"]
    summary_b_data["first.mp4"] = video_summary(clips=["clip-b-1"])
    summary_b_data["third.mp4"] = video_summary(clips=["clip-b-3"])

    specs = build_video_comparison_specs(
        output_a,
        output_b,
        OutputSummary.from_json_dict(summary_a_data),
        OutputSummary.from_json_dict(summary_b_data),
        video_limit=2,
    )

    assert specs == (
        VideoComparisonSpec(
            video_key="first.mp4",
            output_a=str(output_a),
            output_b=str(output_b),
            clips_a=("clip-a-1",),
            clips_b=("clip-b-1",),
        ),
        VideoComparisonSpec(
            video_key="second.mp4",
            output_a=str(output_a),
            output_b=str(output_b),
            clips_a=("clip-a-2",),
            clips_b=(),
        ),
    )


def test_build_video_comparison_specs_selects_exact_selected_video_key(tmp_path: Path) -> None:
    """Exact selected-video-key selection builds one spec matched across both outputs."""
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    summary_a_data = summary()
    del summary_a_data["video.mp4"]
    summary_a_data["first.mp4"] = video_summary(clips=["clip-a-1"])
    summary_a_data["target.mp4"] = video_summary(clips=["clip-a-target"])
    summary_b_data = summary()
    del summary_b_data["video.mp4"]
    summary_b_data["target.mp4"] = video_summary(clips=["clip-b-target"])
    summary_b_data["third.mp4"] = video_summary(clips=["clip-b-3"])

    specs = build_video_comparison_specs(
        output_a,
        output_b,
        OutputSummary.from_json_dict(summary_a_data),
        OutputSummary.from_json_dict(summary_b_data),
        selected_video_key="target.mp4",
    )

    assert specs == (
        VideoComparisonSpec(
            video_key="target.mp4",
            output_a=str(output_a),
            output_b=str(output_b),
            clips_a=("clip-a-target",),
            clips_b=("clip-b-target",),
        ),
    )


def test_build_video_comparison_specs_rejects_unknown_exact_selected_video_key(tmp_path: Path) -> None:
    """Exact selected-video-key selection fails when the key is absent from both summaries."""
    with pytest.raises(ValueError, match="selected_video_key is not present"):
        build_video_comparison_specs(
            tmp_path / "output-a",
            tmp_path / "output-b",
            _output_summary(),
            _output_summary(),
            selected_video_key="missing.mp4",
        )


def test_build_clip_comparison_specs_expands_shared_and_side_only_clips(tmp_path: Path) -> None:
    """Clip specs preserve video grouping while exposing side-specific clip presence."""
    video_specs = (
        VideoComparisonSpec(
            video_key="video.mp4",
            output_a=str(tmp_path / "output-a"),
            output_b=str(tmp_path / "output-b"),
            clips_a=("clip-shared", "clip-a-only"),
            clips_b=("clip-shared", "clip-b-only"),
        ),
    )

    clip_specs = build_clip_comparison_specs(video_specs)

    assert clip_specs == (
        ClipComparisonSpec(
            video_key="video.mp4",
            clip_id="clip-a-only",
            output_a=str(tmp_path / "output-a"),
            output_b=str(tmp_path / "output-b"),
            in_a=True,
            in_b=False,
        ),
        ClipComparisonSpec(
            video_key="video.mp4",
            clip_id="clip-b-only",
            output_a=str(tmp_path / "output-a"),
            output_b=str(tmp_path / "output-b"),
            in_a=False,
            in_b=True,
        ),
        ClipComparisonSpec(
            video_key="video.mp4",
            clip_id="clip-shared",
            output_a=str(tmp_path / "output-a"),
            output_b=str(tmp_path / "output-b"),
            in_a=True,
            in_b=True,
        ),
    )


def test_build_clip_comparison_specs_skips_empty_video_clip_rows(tmp_path: Path) -> None:
    """Videos without clips do not produce clip-level feature work."""
    video_specs = (
        VideoComparisonSpec(
            video_key="empty.mp4",
            output_a=str(tmp_path / "output-a"),
            output_b=str(tmp_path / "output-b"),
            clips_a=(),
            clips_b=(),
        ),
    )

    clip_specs = build_clip_comparison_specs(video_specs)

    assert clip_specs == ()


def test_clip_comparison_spec_json_round_trip(tmp_path: Path) -> None:
    """Clip specs round-trip through JSON-compatible Ray Data rows."""
    spec = ClipComparisonSpec(
        video_key="video.mp4",
        clip_id="clip-a",
        output_a=str(tmp_path / "output-a"),
        output_b=str(tmp_path / "output-b"),
        in_a=True,
        in_b=False,
    )

    assert ClipComparisonSpec.from_json_dict(spec.to_json_dict()) == spec


def test_caption_comparator_builds_clip_feature_plan(tmp_path: Path) -> None:
    """Caption comparator plans selected caption clips as explicit clip feature work."""
    specs = build_video_comparison_specs(
        tmp_path / "output-a", tmp_path / "output-b", _output_summary(), _output_summary()
    )
    context = FeatureComparisonContext(
        output_a=tmp_path / "output-a",
        output_b=tmp_path / "output-b",
        summary_a=_output_summary(),
        summary_b=_output_summary(),
        profile_name="profile-a",
        specs=specs,
    )

    plan = CaptionFeatureComparator().build_plan(context)

    assert isinstance(plan, ClipFeaturePlan)
    assert plan.feature_name == "captions"
    assert plan.clip_specs == (
        ClipComparisonSpec(
            video_key="video.mp4",
            clip_id="clip-a",
            output_a=str(tmp_path / "output-a"),
            output_b=str(tmp_path / "output-b"),
            in_a=True,
            in_b=True,
        ),
    )
    assert plan.load_worker_constructor_kwargs["profile_name"] == "profile-a"


def test_caption_comparator_resolves_no_caption_feature_plan(tmp_path: Path) -> None:
    """No-caption outputs resolve without scheduling artifact work."""
    summary_value = OutputSummary.from_json_dict(
        summary(
            **{
                "video.mp4": video_summary(clips=["clip-a"]),
                "total_num_clips_with_caption": 0,
                "total_num_caption_windows": 0,
            }
        )
    )
    specs = build_video_comparison_specs(tmp_path / "output-a", tmp_path / "output-b", summary_value, summary_value)
    context = FeatureComparisonContext(
        output_a=tmp_path / "output-a",
        output_b=tmp_path / "output-b",
        summary_a=summary_value,
        summary_b=summary_value,
        profile_name="profile-a",
        specs=specs,
    )

    plan = CaptionFeatureComparator().build_plan(context)

    assert isinstance(plan, ResolvedFeaturePlan)
    assert plan.feature_name == "captions"
    assert plan.result.comparison.status == "passed"


def test_compare_features_runs_injected_feature_plans(tmp_path: Path) -> None:
    """Feature comparison dispatches resolved and clip plans without knowing feature-specific code."""
    result = compare_features(
        tmp_path / "output-a",
        tmp_path / "output-b",
        _output_summary(),
        _output_summary(),
        feature_planners=(_FakeResolvedComparator(), _FakeClipComparator()),
    )

    assert result.feature_comparisons["fake_resolved"].metrics == {"planned_videos": 2}
    assert result.feature_comparisons["fake_clip"].metrics == {"rows": 2}


def test_compare_features_rejects_duplicate_feature_plan_names(tmp_path: Path) -> None:
    """Feature plan names are report keys and must not collide."""
    with pytest.raises(ValueError, match="Duplicate feature plan names"):
        compare_features(
            tmp_path / "output-a",
            tmp_path / "output-b",
            _output_summary(),
            _output_summary(),
            feature_planners=(_FakeResolvedComparator(), _FakeResolvedComparator()),
        )


def test_compare_features_rejects_duplicate_feature_result_names(tmp_path: Path) -> None:
    """Mapped feature results must not overwrite earlier feature results."""
    with pytest.raises(ValueError, match="Duplicate feature results"):
        compare_features(
            tmp_path / "output-a",
            tmp_path / "output-b",
            _output_summary(),
            _output_summary(),
            feature_planners=(_FakeResolvedComparator(), _FakeMappedResolvedComparator()),
        )


def test_compare_features_rejects_nonserializable_load_group_kwargs(tmp_path: Path) -> None:
    """Load grouping requires stable constructor identities for custom non-JSON kwargs."""
    with pytest.raises(TypeError, match="set load_group_id"):
        compare_features(
            tmp_path / "output-a",
            tmp_path / "output-b",
            _output_summary(),
            _output_summary(),
            feature_planners=(_FakeNonSerializableClipComparator(),),
        )


def test_compare_features_accepts_explicit_load_group_id_for_nonserializable_kwargs(tmp_path: Path) -> None:
    """Custom planners can provide an explicit stable load group identity."""
    result = compare_features(
        tmp_path / "output-a",
        tmp_path / "output-b",
        _output_summary(),
        _output_summary(),
        feature_planners=(_FakeNonSerializableClipComparator(load_group_id="fake-loader"),),
    )

    assert result.feature_comparisons["fake_nonserializable_clip"].metrics == {"rows": 0}


def test_load_group_identity_preserves_container_types() -> None:
    """Load grouping must not collapse distinct Python container types."""
    assert _canonical_load_worker_constructor_kwargs({"marker": [1, 2]}) != _canonical_load_worker_constructor_kwargs(
        {"marker": (1, 2)}
    )


def test_compare_features_caps_ray_compute_size_to_clip_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ray Data actor/task pool sizes do not exceed the planned clip row count."""
    actor_sizes: list[int] = []
    task_sizes: list[int] = []
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.compare_features.ActorPoolStrategy",
        lambda size: actor_sizes.append(size) or SimpleNamespace(size=size),
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.compare_features.TaskPoolStrategy",
        lambda size: task_sizes.append(size) or SimpleNamespace(size=size),
    )

    compare_features(
        tmp_path / "output-a",
        tmp_path / "output-b",
        _output_summary(),
        _output_summary(),
        feature_planners=(
            CaptionFeatureComparator(),
            ScoreFeatureComparator(AESTHETIC_SCORE_FEATURE_NAME),
            ScoreFeatureComparator(MOTION_SCORE_FEATURE_NAME),
        ),
        workers_per_node=32,
    )

    assert actor_sizes == [1, 2]
    assert task_sizes == [1, 2, 2]


@pytest.mark.parametrize(
    ("ray_config", "message"),
    [
        pytest.param({"workers_per_node": 0}, "workers_per_node must be greater than 0", id="workers"),
        pytest.param({"cpus_per_worker": 0}, "cpus_per_worker must be greater than 0", id="cpus"),
    ],
)
def test_compare_features_rejects_invalid_ray_execution_config(
    tmp_path: Path,
    ray_config: dict[str, int],
    message: str,
) -> None:
    """Ray Data execution settings must be positive."""
    with pytest.raises(ValueError, match=message):
        compare_features(
            tmp_path / "output-a",
            tmp_path / "output-b",
            _output_summary(),
            _output_summary(),
            feature_planners=(
                CaptionFeatureComparator(),
                ScoreFeatureComparator(AESTHETIC_SCORE_FEATURE_NAME),
                ScoreFeatureComparator(MOTION_SCORE_FEATURE_NAME),
            ),
            **ray_config,
        )


def test_compare_features_loads_metadata_once_per_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ray Data rows load metadata once and share it with feature comparators."""
    read_paths: list[str] = []

    def fake_read_json_object(path: object, *, client_params: object) -> dict[str, object]:
        _ = client_params
        read_paths.append(str(path))
        return {
            "span_uuid": Path(str(path)).stem,
            "windows": [
                {
                    "start_frame": 0,
                    "end_frame": 30,
                    "caption_status": "success",
                    "qwen_caption": "caption",
                }
            ],
        }

    summary_value = OutputSummary.from_json_dict(
        summary(
            **{
                "video.mp4": video_summary(clips=["clip-a"])
                | {
                    "num_clips_with_caption": 1,
                    "num_caption_windows": 1,
                },
                "total_num_clips_with_caption": 1,
                "total_num_caption_windows": 1,
            }
        )
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    result = compare_features(
        tmp_path / "output-a",
        tmp_path / "output-b",
        summary_value,
        summary_value,
        feature_planners=(
            CaptionFeatureComparator(),
            ScoreFeatureComparator(AESTHETIC_SCORE_FEATURE_NAME),
            ScoreFeatureComparator(MOTION_SCORE_FEATURE_NAME),
        ),
    )

    assert result.issues == ()
    assert result.feature_comparisons["captions"].status == "passed"
    assert result.feature_comparisons[AESTHETIC_SCORE_FEATURE_NAME].status == "skipped"
    assert result.feature_comparisons[MOTION_SCORE_FEATURE_NAME].status == "skipped"
    assert read_paths == [
        str(tmp_path / "output-a" / "metas" / "v0" / "clip-a.json"),
        str(tmp_path / "output-b" / "metas" / "v0" / "clip-a.json"),
    ]


def test_compare_features_skips_no_caption_video_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caption artifact work only loads clips for selected videos with caption records."""
    read_paths: list[str] = []

    def fake_read_json_object(path: object, *, client_params: object) -> dict[str, object]:
        _ = client_params
        read_paths.append(str(path))
        return {
            "span_uuid": Path(str(path)).stem,
            "windows": [
                {
                    "start_frame": 0,
                    "end_frame": 30,
                    "caption_status": "success",
                    "qwen_caption": "caption",
                }
            ],
        }

    summary_value = OutputSummary.from_json_dict(
        summary(
            num_processed_videos=2,
            **{
                "no-caption.mp4": video_summary(clips=["clip-no-caption"]),
                "video.mp4": video_summary(clips=["clip-caption"])
                | {
                    "num_clips_with_caption": 1,
                    "num_caption_windows": 1,
                },
                "total_num_clips_with_caption": 1,
                "total_num_caption_windows": 1,
            },
        )
    )
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    result = compare_features(
        tmp_path / "output-a",
        tmp_path / "output-b",
        summary_value,
        summary_value,
        feature_planners=(CaptionFeatureComparator(),),
    )

    assert result.issues == ()
    assert result.feature_comparisons["captions"].metrics["videos_with_captions_a"] == 1
    assert read_paths == [
        str(tmp_path / "output-a" / "metas" / "v0" / "clip-caption.json"),
        str(tmp_path / "output-b" / "metas" / "v0" / "clip-caption.json"),
    ]


@pytest.mark.parametrize(
    ("selector_kwargs", "expected_clip_id"),
    [
        pytest.param({"video_limit": 1}, "clip-first", id="limit"),
        pytest.param({"selected_video_key": "second.mp4"}, "clip-second", id="selected-video-key"),
    ],
)
def test_compare_features_selector_scopes_caption_expected_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selector_kwargs: dict[str, int | str],
    expected_clip_id: str,
) -> None:
    """Caption selectors validate counts for selected videos only."""
    read_paths: list[str] = []

    def fake_read_json_object(path: object, *, client_params: object) -> dict[str, object]:
        _ = client_params
        read_paths.append(str(path))
        return {
            "span_uuid": Path(str(path)).stem,
            "windows": [
                {
                    "start_frame": 0,
                    "end_frame": 30,
                    "caption_status": "success",
                    "qwen_caption": "caption",
                }
            ],
        }

    summary_data = summary(
        num_processed_videos=2,
        total_num_clips_with_caption=2,
        total_num_caption_windows=2,
    )
    del summary_data["video.mp4"]
    summary_data["first.mp4"] = video_summary(clips=["clip-first"]) | {
        "num_clips_with_caption": 1,
        "num_caption_windows": 1,
    }
    summary_data["second.mp4"] = video_summary(clips=["clip-second"]) | {
        "num_clips_with_caption": 1,
        "num_caption_windows": 1,
    }
    summary_value = OutputSummary.from_json_dict(summary_data)
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.output_comparison.video_artifacts._read_json_object",
        fake_read_json_object,
    )

    result = compare_features(
        tmp_path / "output-a",
        tmp_path / "output-b",
        summary_value,
        summary_value,
        feature_planners=(
            CaptionFeatureComparator(),
            ScoreFeatureComparator(AESTHETIC_SCORE_FEATURE_NAME),
            ScoreFeatureComparator(MOTION_SCORE_FEATURE_NAME),
        ),
        **selector_kwargs,
    )

    assert result.issues == ()
    assert result.feature_comparisons["captions"].status == "passed"
    assert result.feature_comparisons["captions"].metrics["clips_with_captions_a"] == 1
    assert result.feature_comparisons[AESTHETIC_SCORE_FEATURE_NAME].status == "skipped"
    assert result.feature_comparisons[MOTION_SCORE_FEATURE_NAME].status == "skipped"
    assert read_paths == [
        str(tmp_path / "output-a" / "metas" / "v0" / f"{expected_clip_id}.json"),
        str(tmp_path / "output-b" / "metas" / "v0" / f"{expected_clip_id}.json"),
    ]
