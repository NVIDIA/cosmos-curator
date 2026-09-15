# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multimodal-caption pipeline-kind adapter."""

from pathlib import Path

from cosmos_curator.next.recipes.multimodal_caption.pipeline_kind import MULTIMODAL_CAPTION_KIND


def test_pipeline_kind_validates_and_renders_public_config(tmp_path: Path) -> None:
    """The lightweight adapter exposes the public config operations."""
    config = tmp_path / "caption.yaml"
    config.write_text(
        f"""schema_version: 1
kind: multimodal-caption
input:
  episodes_lance_uri: {tmp_path / "episodes.lance"}
context:
  variant: video_only
  camera_key: front_wide
""",
        encoding="utf-8",
    )

    assert MULTIMODAL_CAPTION_KIND.validate(config, ()) == {"ok": True}
    assert '"kind": "multimodal-caption"' in MULTIMODAL_CAPTION_KIND.render(config, ())
    assert MULTIMODAL_CAPTION_KIND.template_payload()["required_fields"][-1] == {
        "path": "context.camera_key",
        "example": "front_wide",
    }
