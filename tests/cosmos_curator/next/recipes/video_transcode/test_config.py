# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed defaults, identities, and lazy CLI registration."""

import json

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.recipes.video_transcode.config import config_template, resolve_config_data
from cosmos_curator.next.recipes.video_transcode.contracts import make_clip_id


def test_defaults_and_cli_template() -> None:
    """Changing the template's media root updates derived paths and preserves execution defaults."""
    from cosmos_curator.pipelines.builtin_pipeline_kinds import BUILTIN_PIPELINE_KINDS  # noqa: PLC0415

    kind = BUILTIN_PIPELINE_KINDS.get("video-transcode")
    config = resolve_config_data(
        yaml.safe_load(kind.template_yaml()), overrides=["input.media_root=s3://test-bucket/actual-videos"]
    )
    assert config.execution.parquet_rows_per_file == 1_000
    assert config.execution.transcode_memory_gib == 4.0
    assert config.execution.publication_memory_gib == 8.0
    assert config.input.spans_lance_uri == f"{config.input.media_root}/lance"
    assert config.output.staging_root_uri == f"{config.input.media_root}/staging/video-transcode"
    assert "spans_per_fragment" not in type(config.execution).model_fields
    assert json.loads(kind.schema_json())["properties"]["kind"]["const"] == "video-transcode"


def test_encoding_identity_and_execution_overrides() -> None:
    """A new encoding creates a different rendition; tuning does not."""
    data = config_template()
    first = resolve_config_data(data)
    tuned = resolve_config_data(
        data,
        overrides=[
            "execution.parquet_rows_per_file=17",
            "execution.ffmpeg_batch_size=3",
            "execution.transcode_memory_gib=1.5",
            "execution.publication_memory_gib=2.0",
        ],
    )
    assert tuned.execution.transcode_memory_gib == 1.5
    assert tuned.execution.publication_memory_gib == 2.0
    assert first.transcode.contract_id == tuned.transcode.contract_id
    second = resolve_config_data(data, overrides=["transcode.video_bitrate=8M"])
    assert make_clip_id("span", first.transcode.contract_id) != make_clip_id("span", second.transcode.contract_id)


@pytest.mark.parametrize(
    "override",
    [
        "execution.parquet_rows_per_file=0",
        "execution.transcode_memory_gib=0",
        "execution.transcode_memory_gib=-1",
        "execution.transcode_memory_gib=.inf",
        "execution.transcode_memory_gib=.nan",
        "execution.publication_memory_gib=0",
        "execution.publication_memory_gib=-1",
        "execution.publication_memory_gib=.inf",
        "execution.publication_memory_gib=.nan",
        "execution.unknown=1",
        "output.staging_root_uri=s3://example-bucket/curated/videos/lance/staging",
    ],
)
def test_invalid_config(override: str) -> None:
    """Reject invalid tuning and staging that could overlap canonical data."""
    with pytest.raises((ValidationError, ValueError)):
        resolve_config_data(config_template(), overrides=[override])
