# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for video-caption pipeline-kind registration and lazy dispatch."""

import json
from pathlib import Path

import pytest
import yaml

from cosmos_curator.next.recipes.video_caption import pipeline
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig
from cosmos_curator.next.recipes.video_caption.pipeline_kind import VIDEO_CAPTION_KIND


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "video-caption",
                "input": {"media_root": str(path.parent / "media")},
                "model": {"variant": "qwen3_8_27b_fp8"},
            }
        ),
        encoding="utf-8",
    )


def test_kind_is_registered_and_exposes_config_surfaces(tmp_path: Path) -> None:
    """The generic CLI discovers all strict video-caption config operations."""
    from cosmos_curator.pipelines.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS  # noqa: PLC0415

    config_path = tmp_path / "caption.yaml"
    _write_config(config_path)

    assert BUILTIN_PIPELINE_KINDS.get("video-caption") is VIDEO_CAPTION_KIND
    assert VIDEO_CAPTION_KIND.validate(config_path, ()) == {"ok": True}
    assert json.loads(VIDEO_CAPTION_KIND.render(config_path, ()))["kind"] == "video-caption"
    assert json.loads(VIDEO_CAPTION_KIND.schema_json())["title"] == "ResolvedVideoCaptionConfig"
    assert VIDEO_CAPTION_KIND.template_payload()["kind"] == "video-caption"
    assert VIDEO_CAPTION_KIND.list_presets() == []


def test_prepared_run_dispatches_resolved_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deferred execution passes a resolved config to the runtime adapter."""
    config_path = tmp_path / "caption.yaml"
    _write_config(config_path)
    seen: list[ResolvedVideoCaptionConfig] = []

    def run_config(config: ResolvedVideoCaptionConfig) -> dict[str, object]:
        seen.append(config)
        return {
            "published_fragments": 3,
            "skipped_complete_fragments": 4,
        }

    monkeypatch.setattr(pipeline, "run_config", run_config)

    output = VIDEO_CAPTION_KIND.prepare_run(config_path, set_overrides=[])()

    assert seen[0].kind == "video-caption"
    assert output.json_payload["published_fragments"] == 3
    assert output.message == "video-caption: 3 fragment(s) published, 4 complete fragment(s) skipped"
