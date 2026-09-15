# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public multimodal-caption configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from cosmos_curator.next.recipes.multimodal_caption.config import config_template, resolve_config_data


def _raw(uri: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "multimodal-caption",
        "input": {"episodes_lance_uri": uri},
        "context": {"variant": "video_only", "camera_key": "front_wide"},
    }


def test_required_context_and_default_workspace_are_resolved(tmp_path: Path) -> None:
    """Required context is retained and staging defaults beside the table."""
    config = resolve_config_data(_raw(str(tmp_path / "episodes.lance")))

    assert config.context.camera_key == "front_wide"
    assert config.context.variant == "video_only"
    assert config.model.variant == "qwen3_8_27b_fp8"
    assert config.output.staging_root_uri == str(tmp_path / "staging" / "multimodal-caption")


@pytest.mark.parametrize(
    "change",
    [
        {"context": {"variant": "video_only"}},
        {"context": {"variant": "video_only", "camera_key": ""}},
        {"context": {"variant": "ego", "camera_key": "front_wide"}},
        {"unknown": True},
    ],
)
def test_required_camera_variant_and_strict_shape(change: dict[str, object], tmp_path: Path) -> None:
    """The public model rejects missing, unsupported, and unknown input."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw.update(change)
    with pytest.raises(ValidationError):
        resolve_config_data(raw)


def test_template_uses_an_explicit_example_camera_without_a_camera_default() -> None:
    """The template labels an explicit camera choice for users to replace."""
    template = config_template()
    assert template["context"] == {"variant": "video_only", "camera_key": "front_wide"}
