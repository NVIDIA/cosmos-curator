# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for robot-action-split config resolution."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.recipes.robot_action_split.config import (
    ResolvedRobotActionSplitConfig,
    resolve_config,
)


def test_shared_dotted_overrides_apply_before_validation(tmp_path: Path) -> None:
    """Robot-action config uses the shared typed override contract."""
    config_path = tmp_path / "robot-action-split.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "robot-action-split",
                "input": {
                    "uris": ["s3://example-bucket/robot-data/dataset/"],
                    "source_dataset": "libero_10",
                },
                "output": {
                    "media_root": "s3://example-bucket/output/",
                    "lance_uri": "s3://example-bucket/output/clips.lance",
                },
            }
        ),
        encoding="utf-8",
    )

    config = resolve_config(
        config_path,
        overrides=["execution.progress=true", "split.max_duration_s=null"],
    )

    assert config.execution.progress is True
    assert config.split.max_duration_s is None


def _minimal_raw(execution: dict[str, object] | None = None, **sections: dict[str, object]) -> dict[str, object]:
    raw: dict[str, object] = {
        "schema_version": 1,
        "kind": "robot-action-split",
        "input": {"uris": ["s3://example-bucket/robot-data/dataset/"], "source_dataset": "libero_10"},
        "output": {"media_root": "s3://example-bucket/output/"},
    }
    for name, overrides in sections.items():
        raw[name] = {**raw.get(name, {}), **overrides}  # type: ignore[dict-item]
    if execution is not None:
        raw["execution"] = execution
    return raw


def test_input_profile_defaults_to_the_execution_profile() -> None:
    """A single-store config names one profile, and both sides use it."""
    config = ResolvedRobotActionSplitConfig.model_validate(_minimal_raw({"storage_profile": "team-gcs-cosmos"}))
    assert (config.input.storage_profile, config.execution.storage_profile) == ("team-gcs-cosmos", "team-gcs-cosmos")


def test_input_profile_defaults_when_execution_absent() -> None:
    """Omitting both profiles reads and writes through the default profile."""
    config = ResolvedRobotActionSplitConfig.model_validate(_minimal_raw())
    assert (config.input.storage_profile, config.execution.storage_profile) == ("default", "default")


def test_input_profile_overrides_the_execution_profile() -> None:
    """Sources in another object store keep their own profile; outputs use the execution profile."""
    raw = _minimal_raw({"storage_profile": "aidot"}, input={"storage_profile": "team-gcs-cosmos"})
    config = ResolvedRobotActionSplitConfig.model_validate(raw)
    assert (config.input.storage_profile, config.execution.storage_profile) == ("team-gcs-cosmos", "aidot")


@pytest.mark.parametrize("profile", ["", None])
def test_an_explicit_empty_input_profile_is_rejected_not_backfilled(profile: str | None) -> None:
    """An explicit bad value is a broken override, never silently replaced by the output profile."""
    with pytest.raises(ValidationError, match="storage_profile"):
        ResolvedRobotActionSplitConfig.model_validate(_minimal_raw(input={"storage_profile": profile}))


@pytest.mark.parametrize(
    "execution",
    [
        {"output_storage_profile": "aidot"},
        {"clips_per_publish_batch": 10},
        {"max_segments_per_batch": 10},
        {"ray_data": False},
        {"storage_attempts": 3},
        {"cut_attempts": 3},
    ],
)
def test_retired_execution_settings_are_rejected(execution: dict[str, object]) -> None:
    """Settings of the retired publication path fail loudly instead of being ignored."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResolvedRobotActionSplitConfig.model_validate(_minimal_raw(execution))


def test_output_locations_derive_from_the_media_root() -> None:
    """The clip table and recovery workspace default beneath the media root."""
    config = ResolvedRobotActionSplitConfig.model_validate(_minimal_raw())
    assert config.output.media_root == "s3://example-bucket/output"
    assert config.output.lance_uri == "s3://example-bucket/output/lance/clips.lance"
    assert config.output.staging_root_uri == "s3://example-bucket/output/staging/robot-action-split"


@pytest.mark.parametrize(
    ("lance_uri", "staging_root_uri"),
    [
        ("s3://example-bucket/output/clips.lance", "s3://example-bucket/output/clips.lance"),
        ("s3://example-bucket/output/clips.lance", "s3://example-bucket/output/clips.lance/staging"),
        ("s3://example-bucket/output/staging/clips.lance", "s3://example-bucket/output/staging"),
    ],
)
def test_staging_must_be_separate_from_the_table(lance_uri: str, staging_root_uri: str) -> None:
    """Recovery files never land inside the Lance table, or the table inside the workspace."""
    raw = _minimal_raw(output={"lance_uri": lance_uri, "staging_root_uri": staging_root_uri})
    with pytest.raises(ValidationError, match="separate directories"):
        ResolvedRobotActionSplitConfig.model_validate(raw)


def test_split_contract_holds_result_settings_and_never_execution_tuning() -> None:
    """Tuning, table and staging locations, views, and limits leave the contract unchanged.

    Split and encoding settings change it, and so does the media root that finished clips reference.
    """
    base = ResolvedRobotActionSplitConfig.model_validate(_minimal_raw())
    tuned = ResolvedRobotActionSplitConfig.model_validate(
        _minimal_raw(
            {"cut_cpus": 1.0, "max_items": 3, "rows_per_checkpoint": 1, "storage_profile": "other"},
            input={"limit": 2},
            output={
                "lance_uri": "s3://example-bucket/tables/clips.lance",
                "staging_root_uri": "s3://example-bucket/staging",
                "views": ["observation.images.main"],
            },
        )
    )
    assert tuned.split_contract == base.split_contract
    for section, change in (
        ("output", {"media_root": "s3://example-bucket/elsewhere/"}),
        ("split", {"min_duration_s": 3.0}),
        ("output", {"video_bitrate": "4M"}),
        ("output", {"action_format": "pickle"}),
        ("input", {"source_dataset": "libero_90"}),
    ):
        changed = ResolvedRobotActionSplitConfig.model_validate(_minimal_raw(**{section: change}))
        assert changed.split_contract != base.split_contract, (section, change)


@pytest.mark.parametrize(("timeout", "valid"), [(60.0, True), (-1.0, True), (0.0, False), (-5.0, False)])
def test_no_progress_timeout_is_positive_or_disabled(timeout: float, *, valid: bool) -> None:
    """A whole-chunk task can outlast Ray's default window; only -1 disables the check."""
    raw = _minimal_raw({"no_progress_timeout_s": timeout})
    if valid:
        assert ResolvedRobotActionSplitConfig.model_validate(raw).execution.no_progress_timeout_s == timeout
    else:
        with pytest.raises(ValidationError, match="no_progress_timeout_s"):
            ResolvedRobotActionSplitConfig.model_validate(raw)


@pytest.mark.parametrize(
    "example",
    [
        "examples/robot-action-split-lerobot.yaml",
        "examples/vitra-epic-split.yaml",
        "examples/slurm/robot-action-split-draco-oci-1-batch01.yaml",
    ],
)
def test_tracked_examples_resolve(example: str) -> None:
    """Published example configs stay runnable against the current schema."""
    repo_root = Path(__file__).resolve().parents[5]
    assert resolve_config(repo_root / example).kind == "robot-action-split"


def test_act2_actions_require_a_registered_dataset() -> None:
    """An unregistered dataset would fail every clip's action encoding, permanently, so it fails validation."""
    with pytest.raises(ValidationError, match="No action-binary spec registered"):
        ResolvedRobotActionSplitConfig.model_validate(_minimal_raw(input={"source_dataset": "unregistered"}))
    pickled = _minimal_raw(input={"source_dataset": "unregistered"}, output={"action_format": "pickle"})
    assert ResolvedRobotActionSplitConfig.model_validate(pickled).input.source_dataset == "unregistered"


@pytest.mark.parametrize(
    ("bitrate", "valid"), [("2M", True), ("2500k", True), ("1.5M", True), ("2 M", False), ("", False)]
)
def test_video_bitrate_is_an_ffmpeg_rate(bitrate: str, *, valid: bool) -> None:
    """A rate ffmpeg cannot parse would fail every re-encode of the run."""
    raw = _minimal_raw(output={"video_bitrate": bitrate})
    if valid:
        assert ResolvedRobotActionSplitConfig.model_validate(raw).output.video_bitrate == bitrate
    else:
        with pytest.raises(ValidationError, match="video_bitrate"):
            ResolvedRobotActionSplitConfig.model_validate(raw)
