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

"""Tests for public multimodal-embedding configuration and pipeline-kind dispatch."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.recipes.multimodal_embedding.config import (
    config_schema_json,
    config_template,
    config_template_payload,
    config_template_yaml,
    resolve_config_data,
)
from cosmos_curator.next.recipes.multimodal_embedding.pipeline_kind import MULTIMODAL_EMBEDDING_KIND


def _config_payload(uri: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "multimodal-embedding",
        "input": {"episodes_lance_uri": uri},
        "video": {"camera_key": "front_wide"},
    }


def test_defaults_resolve_to_a_complete_video_embedding_config(tmp_path: Path) -> None:
    """The minimal public shape picks one pinned model and selection policy."""
    config = resolve_config_data(_config_payload(str(tmp_path / "episodes.lance")))

    assert config.model.variant == "cosmos_embed1_336p"
    assert config.output.result_key == "default"
    assert config.selection.max_episode_duration_s == 60.0
    assert config.selection.retry_errors is False
    assert config.execution.media_attempts == 3
    assert config.execution.decode_cpus == 1
    assert config.execution.decode_memory_gib == 2.0
    assert config.execution.media_concurrency == config.execution.decode_concurrency == "auto"
    assert config.execution.inference_batch_size == 4
    assert config.output.staging_root_uri == str(tmp_path / "staging" / "multimodal-embedding")


def test_null_duration_limit_is_rejected(tmp_path: Path) -> None:
    """Every run has a finite positive eligibility ceiling."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload["selection"] = {"max_episode_duration_s": None}
    with pytest.raises(ValidationError, match="max_episode_duration_s"):
        resolve_config_data(config_payload)


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "embeddings"},
        {"video": {"camera_key": ""}},
        {"video": {"camera_key": " front_wide"}},
        {"model": {"variant": "internvideo2_mm"}},
        {"selection": {"max_episode_duration_s": 0.0}},
        {"execution": {"media_attempts": 0}},
        {"execution": {"decode_cpus": 0}},
        {"execution": {"decode_cpus": 1.5}},
        {"execution": {"decode_memory_gib": 0}},
        {"execution": {"decode_memory_gib": float("inf")}},
        {"execution": {"resource_ready_timeout_s": 120.0}},
        {"unknown": True},
    ],
)
def test_config_is_strict_about_its_public_contract(change: dict[str, object], tmp_path: Path) -> None:
    """No legacy model, malformed camera, or unknown setting is silently accepted."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload.update(change)

    with pytest.raises(ValidationError):
        resolve_config_data(config_payload)


@pytest.mark.parametrize("field", ["media_concurrency", "decode_concurrency", "inference_concurrency"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "4", "unlimited", None])
def test_concurrency_requires_auto_or_a_positive_integer(field: str, value: object, tmp_path: Path) -> None:
    """Stage limits reject invalid values rather than silently changing admission."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload["execution"] = {field: value}
    with pytest.raises(ValidationError, match=field):
        resolve_config_data(config_payload)


@pytest.mark.parametrize("field", ["media_concurrency", "decode_concurrency", "inference_concurrency"])
@pytest.mark.parametrize("value", ["auto", 1, 8])
def test_stage_concurrency_accepts_operator_overrides(field: str, value: object, tmp_path: Path) -> None:
    """CLI overrides resolve through the same strict stage concurrency contract."""
    resolved = resolve_config_data(
        _config_payload(str(tmp_path / "episodes.lance")), overrides=[f"execution.{field}={value}"]
    )
    assert getattr(resolved.execution, field) == value


@pytest.mark.parametrize("result_key", ["Default", "9result", "has-dash", "a" * 49, None, 7])
def test_result_key_is_safe_to_use_as_a_field_name_segment(result_key: object, tmp_path: Path) -> None:
    """Result namespaces share the caption recipe's deterministic key grammar."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload["output"] = {"result_key": result_key}

    with pytest.raises(ValidationError):
        resolve_config_data(config_payload)


def test_generated_config_surfaces_are_consistent() -> None:
    """Template, schema, and payload expose the same runnable contract."""
    template = config_template()
    assert template["video"] == {"camera_key": "front_wide"}
    assert template["model"] == {"variant": "cosmos_embed1_336p"}
    assert template["selection"]["max_episode_duration_s"] == 60.0  # type: ignore[index]
    assert template["execution"]["inference_batch_size"] == 4  # type: ignore[index]
    assert template["execution"]["media_concurrency"] == template["execution"]["decode_concurrency"] == "auto"  # type: ignore[index]
    annotated = config_template_yaml()
    parsed = yaml.safe_load("\n".join(line for line in annotated.splitlines() if not line.startswith("#")))
    assert resolve_config_data(parsed).model_dump(mode="json") == resolve_config_data(template).model_dump(mode="json")
    required = {item["path"] for item in config_template_payload()["required_fields"]}
    assert {"input.episodes_lance_uri", "video.camera_key"} <= required
    schema = json.loads(config_schema_json())
    assert "cosmos_embed1_336p" in json.dumps(schema)


@pytest.mark.parametrize("template_format", ["yaml", "json"])
@pytest.mark.parametrize("storage", ["local", "s3"])
def test_template_staging_default_follows_edited_input(template_format: str, storage: str, tmp_path: Path) -> None:
    """Editing only the input cannot retain an example workspace in either template format."""
    template = (
        yaml.safe_load(config_template_yaml())
        if template_format == "yaml"
        else json.loads(json.dumps(config_template_payload()["config"]))
    )
    assert "staging_root_uri" not in template["output"]
    parent = "s3://real-bucket/real-run" if storage == "s3" else str(tmp_path / "real-run")
    template["input"]["episodes_lance_uri"] = f"{parent}/episodes.lance"

    resolved = resolve_config_data(template)

    assert resolved.output.staging_root_uri == f"{parent}/staging/multimodal-embedding"


def test_template_accepts_explicit_staging_override(tmp_path: Path) -> None:
    """Operators can still choose a separate workspace after generating a template."""
    template = config_template()
    staging = str(tmp_path / "custom-staging")
    resolved = resolve_config_data(template, overrides=[f"output.staging_root_uri={staging}"])

    assert resolved.output.staging_root_uri == staging


@pytest.mark.parametrize("variant", ["cosmos_embed1_224p", "cosmos_embed1_336p", "cosmos_embed1_448p"])
def test_cosmos_embed1_variants_are_explicit_model_options(variant: str, tmp_path: Path) -> None:
    """Changing the default does not remove the eight-frame Cosmos path."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload["model"] = {"variant": variant}
    assert resolve_config_data(config_payload).model.variant == variant


def test_pipeline_kind_validates_and_renders_public_config(tmp_path: Path) -> None:
    """The lightweight adapter exposes config operations without loading runtime dependencies."""
    config = tmp_path / "embedding.yaml"
    config.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "kind: multimodal-embedding",
                "input:",
                f"  episodes_lance_uri: {tmp_path / 'episodes.lance'}",
                "video:",
                "  camera_key: front_wide",
                "output:",
                "  result_key: default",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert MULTIMODAL_EMBEDDING_KIND.validate(config, ()) == {"ok": True}
    assert '"kind": "multimodal-embedding"' in MULTIMODAL_EMBEDDING_KIND.render(config, ())
    assert {"path": "video.camera_key", "example": "front_wide"} in MULTIMODAL_EMBEDDING_KIND.template_payload()[
        "required_fields"
    ]
