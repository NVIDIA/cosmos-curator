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
                    "source_dataset": "dataset",
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


def _minimal_raw(execution: dict[str, object] | None = None) -> dict[str, object]:
    raw: dict[str, object] = {
        "schema_version": 1,
        "kind": "robot-action-split",
        "input": {"uris": ["s3://example-bucket/robot-data/dataset/"], "source_dataset": "dataset"},
        "output": {
            "media_root": "s3://example-bucket/output/",
            "lance_uri": "s3://example-bucket/output/clips.lance",
        },
    }
    if execution is not None:
        raw["execution"] = execution
    return raw


def test_output_storage_profile_defaults_to_input_profile() -> None:
    """A single-store config keeps working: output inherits the input profile."""
    config = ResolvedRobotActionSplitConfig.model_validate(_minimal_raw({"storage_profile": "team-gcs-cosmos"}))

    assert config.execution.output_storage_profile == "team-gcs-cosmos"


def test_output_storage_profile_defaults_when_execution_absent() -> None:
    """Omitting the execution block entirely still resolves both profiles."""
    config = ResolvedRobotActionSplitConfig.model_validate(_minimal_raw())

    assert config.execution.storage_profile == "default"
    assert config.execution.output_storage_profile == "default"


def test_output_storage_profile_overrides_input_profile() -> None:
    """Input and output may live in different object stores, each with its own endpoint."""
    config = ResolvedRobotActionSplitConfig.model_validate(
        _minimal_raw({"storage_profile": "team-gcs-cosmos", "output_storage_profile": "aidot"})
    )

    assert config.execution.storage_profile == "team-gcs-cosmos"
    assert config.execution.output_storage_profile == "aidot"


def test_empty_output_storage_profile_is_rejected_not_backfilled() -> None:
    """An explicit empty profile is a broken override, not an omission.

    Backfilling it from ``storage_profile`` would silently write output to the input store -- for a
    GCS-in/S3-out config, the clips would land in the source bucket instead of failing the min_length check.
    """
    with pytest.raises(ValidationError):
        ResolvedRobotActionSplitConfig.model_validate(
            _minimal_raw({"storage_profile": "team-gcs-cosmos", "output_storage_profile": ""})
        )


def test_null_output_storage_profile_is_rejected_not_backfilled() -> None:
    """``output_storage_profile: null`` is likewise an explicit bad value, not an omission."""
    with pytest.raises(ValidationError):
        ResolvedRobotActionSplitConfig.model_validate(
            _minimal_raw({"storage_profile": "team-gcs-cosmos", "output_storage_profile": None})
        )
