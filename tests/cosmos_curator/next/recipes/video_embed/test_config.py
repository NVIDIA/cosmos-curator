# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict config, model contracts, and lazy CLI integration."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.recipes.video_embed import config as config_module
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec, resolve_model_spec
from cosmos_curator.next.recipes.video_embed.pipeline_kind import VIDEO_EMBED_KIND


@pytest.mark.parametrize(("variant", "dimension"), [("224p", 256), ("336p", 768), ("448p", 768)])
def test_model_variants(variant: str, dimension: int) -> None:
    """Each variant names its pinned processor, dimension, and independent field set."""
    config = config_module.resolve_config_data(config_module.config_template(), overrides=[f"model.variant={variant}"])
    spec = resolve_model_spec(config.model.variant)
    assert spec.dimension == dimension
    assert spec.fields.field(spec.vector_field).type.list_size == dimension
    assert spec.contract["processor"]["overrides"] == {}
    assert spec.vector_field == f"embedding_cosmos_embed1_{variant}_v1"
    assert spec.contract["decode"]["backend"] == "pynvvideocodec"
    assert spec.contract["processor"]["device"] == "cuda"
    assert len(spec.revision) == 40


@pytest.mark.parametrize(
    "override",
    [
        "model.variant=internvideo2",
        "model.fps=2",
        "model.frames=16",
        "model.precision=float16",
        "execution.inference_batch_size=0",
        "execution.inference_concurrency=0",
        "execution.inference_concurrency=false",
        "execution.publication_memory_gib=0",
        "execution.publication_memory_gib=-1",
        "execution.publication_memory_gib=.inf",
        "execution.publication_memory_gib=.nan",
        "execution.decode_concurrency=2",
        "execution.decode_cpus=1",
        "execution.decode_memory_gib=2.0",
        "execution.media_concurrency=2",
        "unknown=true",
        "schema_version=2",
        "input.media_root=https://example.com/clips",
        "output.staging_root_uri=s3://example-bucket/curated/videos",
    ],
)
def test_invalid_config(override: str) -> None:
    """Reject unknown knobs, unsupported models, coercion, and unsafe path overlap."""
    with pytest.raises((ValueError, ValidationError)):
        config_module.resolve_config_data(config_module.config_template(), overrides=[override])


def test_template_and_lazy_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: ModelSpec) -> None:
    """CLI tooling and run dispatch use the same resolved configuration."""
    from cosmos_curator.next.recipes.video_embed import pipeline  # noqa: PLC0415
    from cosmos_curator.pipelines.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS  # noqa: PLC0415

    template = config_module.config_template()
    assert template["execution"]["publication_memory_gib"] == 8.0
    assert yaml.safe_load(VIDEO_EMBED_KIND.template_yaml()) == template
    path = tmp_path / "embed.yaml"
    path.write_text(yaml.safe_dump(template), encoding="utf-8")
    kind = BUILTIN_PIPELINE_KINDS.get("video-embed")
    assert kind.validate(path, ()) == {"ok": True}
    rendered = json.loads(
        kind.render(path, ("execution.inference_batch_size=2", "execution.publication_memory_gib=1.5"))
    )
    assert rendered["execution"]["inference_batch_size"] == 2
    assert rendered["execution"]["publication_memory_gib"] == 1.5
    assert rendered["input"]["clips_lance_uri"].endswith("/lance")
    assert kind.list_presets() == []
    assert json.loads(kind.schema_json())["properties"]["kind"]["const"] == "video-embed"
    assert config_module.resolve_config_data(rendered).model.variant == spec.variant
    monkeypatch.setattr(
        pipeline,
        "run_config",
        lambda config: {
            "input_rows": 4,
            "selected_rows": config.execution.inference_batch_size,
            "remaining_fragments": 1,
        },
    )
    run = kind.prepare_run(path, set_overrides=["execution.inference_batch_size=3"])
    output = run()
    assert output.json_payload == {"input_rows": 4, "selected_rows": 3, "remaining_fragments": 1}
    assert output.message == "video-embed: published results for 3 of 4 input row(s); 1 fragment(s) pending"
