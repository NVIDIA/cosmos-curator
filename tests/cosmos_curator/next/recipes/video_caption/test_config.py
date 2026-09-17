# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the strict, versioned video-caption configuration."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.recipes.video_caption.config import (
    VideoCaptionExecutionConfig,
    config_template,
    config_template_yaml,
    resolve_config,
    resolve_config_data,
)


def _minimal(media_root: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "video-caption",
        "input": {"media_root": media_root},
        "model": {"variant": "qwen3_8_27b_fp8"},
    }


def test_defaults_derive_contract_paths_for_s3() -> None:
    """S3 defaults remain under the normalized video-transcode media root."""
    config = resolve_config_data(_minimal("s3://example-bucket/curated/video-transcode/"))

    assert config.input.media_root == "s3://example-bucket/curated/video-transcode"
    assert config.input.clips_lance_uri == "s3://example-bucket/curated/video-transcode/lance"
    assert config.output.staging_root_uri == "s3://example-bucket/curated/video-transcode/staging/video-caption"
    assert config.execution.storage_profile == "default"
    assert config.execution.inference_concurrency == "auto"
    assert config.execution.media_concurrency == "auto"
    assert config.execution.inference_batch_size == 32
    assert config.execution.max_concurrent_batches == 8
    assert config.execution.parquet_rows_per_file == 4_096


def test_defaults_derive_absolute_local_paths(tmp_path: Path) -> None:
    """Local defaults resolve to absolute shared-filesystem paths."""
    config = resolve_config_data(_minimal(str(tmp_path / "media")))

    assert config.input.media_root == str((tmp_path / "media").resolve())
    assert config.input.clips_lance_uri == str((tmp_path / "media/lance").resolve())
    assert config.output.staging_root_uri == str((tmp_path / "media/staging/video-caption").resolve())


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": 2},
        {"kind": "video_caption"},
        {"unexpected": True},
        {"model": {"variant": "Qwen/Qwen3.8-27B-FP8"}},
        {"execution": {"inference_batch_size": "32"}},
        {"execution": {"inference_concurrency": 0}},
        {"execution": {"media_concurrency": 0}},
        {"input": {"media_root": "https://example.com/clips"}},
    ],
)
def test_config_rejects_non_contract_values(update: dict[str, object]) -> None:
    """Strict v1 config rejects aliases, extras, coercion, and arbitrary models."""
    raw = _minimal("s3://example-bucket/media") | update
    if "model" not in update:
        raw["model"] = {"variant": "qwen3_8_27b_fp8"}
    if "input" not in update:
        raw["input"] = {"media_root": "s3://example-bucket/media"}

    with pytest.raises(ValidationError):
        resolve_config_data(raw)


def test_bf16_variant_and_execution_overrides_resolve() -> None:
    """The second pinned variant and dotted execution overrides are supported."""
    config = resolve_config_data(
        _minimal("s3://example-bucket/media"),
        overrides=[
            "model.variant=qwen3_8_27b",
            "execution.inference_concurrency=4",
            "execution.media_concurrency=8",
            "execution.progress=true",
        ],
    )

    assert config.model.variant == "qwen3_8_27b"
    assert config.execution.inference_concurrency == 4
    assert config.execution.media_concurrency == 8
    assert config.execution.progress is True


def test_yaml_loader_and_explicit_locations(tmp_path: Path) -> None:
    """YAML accepts explicit local Lance and recovery locations."""
    config_path = tmp_path / "caption.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _minimal("s3://example-bucket/media")
            | {
                "input": {
                    "media_root": "s3://example-bucket/media",
                    "clips_lance_uri": str(tmp_path / "clips.lance"),
                },
                "output": {"staging_root_uri": str(tmp_path / "recovery")},
            }
        ),
        encoding="utf-8",
    )

    config = resolve_config(config_path)

    assert config.input.clips_lance_uri == str((tmp_path / "clips.lance").resolve())
    assert config.output.staging_root_uri == str((tmp_path / "recovery").resolve())


def test_template_is_complete_and_round_trips() -> None:
    """The discoverable template exposes every execution default and validates."""
    template = config_template()

    assert set(template["execution"]) == set(VideoCaptionExecutionConfig.model_fields)
    assert template["model"]["variant"] == "qwen3_8_27b_fp8"
    assert template["execution"]["media_concurrency"] == "auto"
    assert yaml.safe_load(config_template_yaml()) == template
    assert resolve_config_data(template).model_dump(mode="json") == template
