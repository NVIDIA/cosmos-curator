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
"""Tests for SplitComparisonConfig (pydantic v2) and its nested policy types."""

import pytest
from pydantic import ValidationError

from cosmos_curator.pipelines.video.split_comparison.config import (
    CaptionPolicy,
    ScoreTolerance,
    SplitComparisonConfig,
    SummaryPolicy,
    VideoIndexPolicy,
    example_default_config,
)


def _config(**overrides: object) -> SplitComparisonConfig:
    """Build a config with placeholder targets so tests only spell out the override under test."""
    return SplitComparisonConfig(output_a="/a", output_b="/b", **overrides)  # type: ignore[arg-type]


# --- construction + defaults --------------------------------------------------------


def test_default_config_constructs_with_expected_policy_defaults() -> None:
    """Construction fills every nested policy with documented defaults."""
    config = _config()

    assert config.profile_name == "default"
    assert config.compare_video_index is True
    assert config.compare_captions is True
    assert config.caption.model_id == "BAAI/bge-small-en-v1.5"
    assert config.caption.min_similarity == 0.85
    assert config.aesthetic.abs_tolerance == ScoreTolerance().abs_tolerance
    assert config.video_index.int_tolerance == 0
    assert config.video_index.float_rtol == 1e-5


def test_default_config_metadata_workers_is_at_least_one() -> None:
    """metadata_workers default scales with CPU count but never falls below 1."""
    assert _config().metadata_workers >= 1


def test_default_report_path_and_format() -> None:
    """Persistence default: a single local JSON report."""
    config = _config()
    assert config.report_path == "report.json"
    assert config.report_format == "json"


def test_video_index_default_metadata_fields_are_a_tuple() -> None:
    """compare_metadata_fields is a tuple so frozen-deep is real."""
    policy = VideoIndexPolicy()
    assert isinstance(policy.compare_metadata_fields, tuple)
    assert "codec_name" in policy.compare_metadata_fields
    assert "width" in policy.compare_metadata_fields


def test_config_overrides_propagate_to_nested_policies() -> None:
    """Custom nested policies replace defaults without affecting other fields."""
    config = _config(
        caption=CaptionPolicy(model_id="intfloat/e5-small-v2", min_similarity=0.9),
        compare_video_index=False,
    )

    assert config.caption.model_id == "intfloat/e5-small-v2"
    assert config.caption.min_similarity == 0.9
    assert config.compare_video_index is False
    assert config.aesthetic.abs_tolerance == ScoreTolerance().abs_tolerance


# --- frozen=True (top-level and nested) ---------------------------------------------


def test_config_is_frozen() -> None:
    """Top-level config is immutable; rebinding raises ValidationError."""
    config = _config()
    with pytest.raises(ValidationError):
        config.profile_name = "other"  # type: ignore[misc]


def test_caption_policy_is_frozen() -> None:
    """Nested policies are frozen too; nested mutation is the classic foot-gun this prevents."""
    policy = CaptionPolicy()
    with pytest.raises(ValidationError):
        policy.model_id = "other"  # type: ignore[misc]


# --- strict=True (no type coercion) --------------------------------------------------


def test_strict_mode_rejects_string_for_int_field() -> None:
    """strict=True: "5" must NOT silently coerce to 5 (would mask config-file typos)."""
    with pytest.raises(ValidationError):
        _config(metadata_workers="5")  # type: ignore[arg-type]


def test_strict_mode_rejects_string_for_bool_field() -> None:
    """strict=True: "true" must NOT silently coerce to True."""
    with pytest.raises(ValidationError):
        _config(compare_video_index="true")  # type: ignore[arg-type]


# --- extra="forbid" (unknown fields rejected) ---------------------------------------


def test_unknown_field_rejected() -> None:
    """A typo in a field name fails at construction instead of being silently ignored."""
    with pytest.raises(ValidationError):
        _config(metadta_workers=8)  # type: ignore[call-arg] -- intentional typo


def test_unknown_field_in_nested_model_rejected() -> None:
    """Same protection on nested policy models."""
    with pytest.raises(ValidationError):
        ScoreTolerance(abs_tolarence=0.5)  # type: ignore[call-arg] -- intentional typo


# --- validators (ge=, le=, min_length=, gt=) ----------------------------------------


def test_negative_tolerance_rejected() -> None:
    """Negative tolerance is nonsense; pydantic enforces ge=0.0."""
    with pytest.raises(ValidationError):
        ScoreTolerance(abs_tolerance=-1.0)


def test_min_similarity_outside_zero_to_one_rejected() -> None:
    """min_similarity is a cosine similarity; out-of-[0,1] is rejected."""
    with pytest.raises(ValidationError):
        CaptionPolicy(min_similarity=1.5)
    with pytest.raises(ValidationError):
        CaptionPolicy(min_similarity=-0.1)


def test_zero_metadata_workers_rejected() -> None:
    """metadata_workers must be ge=1; zero would mean "no actors" which fails Stage 1."""
    with pytest.raises(ValidationError):
        _config(metadata_workers=0)


def test_zero_metadata_cpus_per_worker_rejected() -> None:
    """metadata_cpus_per_worker must be gt=0; 0 makes no sense to Ray."""
    with pytest.raises(ValidationError):
        _config(metadata_cpus_per_worker=0.0)


def test_metadata_cpus_per_worker_accepts_fractional_values() -> None:
    """Fractional CPU reservations (e.g. 0.25) are valid -- this is the I/O-pack pattern."""
    config = _config(metadata_cpus_per_worker=0.25)
    assert config.metadata_cpus_per_worker == 0.25


def test_zero_metadata_batch_size_rejected() -> None:
    """metadata_batch_size must be ge=1; zero would make the per-stage block math divide by zero."""
    with pytest.raises(ValidationError):
        _config(metadata_batch_size=0)


def test_zero_video_index_batch_size_rejected() -> None:
    """video_index_batch_size must be ge=1; zero would make the per-stage block math divide by zero."""
    with pytest.raises(ValidationError):
        _config(video_index_batch_size=0)


def test_zero_clip_limit_rejected() -> None:
    """clip_limit must be ge=1 when set; 0 would mean "compare nothing" -- omit instead."""
    with pytest.raises(ValidationError):
        _config(clip_limit=0)


def test_negative_clip_limit_rejected() -> None:
    """Negative clip_limit is rejected at validation time."""
    with pytest.raises(ValidationError):
        _config(clip_limit=-1)


def test_clip_limit_none_means_no_filter() -> None:
    """Omitted clip_limit leaves the field at None (no cap)."""
    assert _config().clip_limit is None


def test_empty_string_for_required_min_length_field_rejected() -> None:
    """profile_name has min_length=1; empty string fails validation."""
    with pytest.raises(ValidationError):
        _config(profile_name="")


def test_empty_output_path_rejected() -> None:
    """output_a / output_b are required and min_length=1; empty string is not a path."""
    with pytest.raises(ValidationError):
        SplitComparisonConfig(output_a="", output_b="/b")


def test_missing_output_a_rejected() -> None:
    """output_a is required (no default); construction without it fails."""
    with pytest.raises(ValidationError):
        SplitComparisonConfig(output_b="/b")  # type: ignore[call-arg]


def test_empty_report_path_rejected() -> None:
    """report_path must be non-empty (min_length=1)."""
    with pytest.raises(ValidationError):
        _config(report_path="")


def test_unknown_report_format_rejected() -> None:
    """report_format only accepts the writer-known values; anything else fails validation."""
    with pytest.raises(ValidationError):
        _config(report_format="parquet")


def test_report_format_lance_accepted() -> None:
    """'lance' is a valid format selection."""
    config = _config(report_format="lance")
    assert config.report_format == "lance"


def test_report_path_accepts_cloud_url_with_any_extension() -> None:
    """report_path is used verbatim -- cloud URLs (and extensionless paths) are fine; format is explicit."""
    config = _config(report_path="s3://bucket/audit", report_format="lance")
    assert config.report_path == "s3://bucket/audit"


# --- JSON round-trip ----------------------------------------------------------------


def test_default_config_round_trips_through_json() -> None:
    """Dump to JSON, load back, equal -- the model.validate_json contract."""
    original = _config(
        compare_captions=False,
        clip_limit=42,
        aesthetic=ScoreTolerance(abs_tolerance=0.01, rel_tolerance=0.02),
        summary=SummaryPolicy(token_count_abs_tolerance=5.0),
    )

    payload = original.model_dump_json()
    reloaded = SplitComparisonConfig.model_validate_json(payload)

    assert reloaded == original


def test_example_default_config_is_valid_and_serializable() -> None:
    """example_default_config() (for --print-default-config) constructs and dumps cleanly."""
    config = example_default_config()
    payload = config.model_dump_json(indent=2)
    # The placeholder strings appear in the output so users see what to replace.
    assert "REPLACE_WITH_OUTPUT_A_PATH" in payload
    assert "REPLACE_WITH_OUTPUT_B_PATH" in payload
    # Round-trip works because the placeholders are non-empty strings.
    assert SplitComparisonConfig.model_validate_json(payload) == config
