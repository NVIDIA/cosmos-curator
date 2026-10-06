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
    MultimodalEmbeddingInputConfig,
    MultimodalEmbeddingOutputConfig,
    ResolvedMultimodalEmbeddingConfig,
    VideoEmbeddingConfig,
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
        "video": {"camera_keys": ["front_wide"]},
    }


def test_defaults_resolve_to_a_complete_video_embedding_config(tmp_path: Path) -> None:
    """The minimal public shape picks one pinned model and selection policy."""
    config = resolve_config_data(_config_payload(str(tmp_path / "episodes.lance")))

    assert config.model.variant == "cosmos_embed1_336p"
    assert config.output.result_key == "default"
    assert config.result_key_for_camera("front_wide") == "default_front_wide"
    assert config.selection.max_episode_duration_s == 60.0
    assert config.selection.retry_errors is False
    assert config.execution.media_attempts == 3
    assert config.execution.decode_cpus == 1
    assert config.execution.decode_memory_gib == 2.0
    assert config.execution.media_concurrency == config.execution.decode_concurrency == "auto"
    assert config.execution.inference_batch_size == 4
    assert config.output.staging_root_uri == str(tmp_path / "staging" / "multimodal-embedding")


@pytest.mark.parametrize("selector", ["single", "list", "all"])
@pytest.mark.parametrize("typed_sections", ["video", "output", "all"])
def test_nested_config_models_preserve_selector_defaults(tmp_path: Path, selector: str, typed_sections: str) -> None:
    """Typed config objects resolve the same tagging and staging defaults as mappings."""
    uri = str(tmp_path / "episodes.lance")
    payload = _config_payload(uri)
    if selector != "single":
        payload["video"] = {"camera_keys": ["front_wide", "rear"] if selector == "list" else "all"}
    payload["output"] = {}
    expected = ResolvedMultimodalEmbeddingConfig.model_validate(payload)
    if typed_sections in {"video", "all"}:
        payload["video"] = VideoEmbeddingConfig.model_validate(payload["video"])
    if typed_sections in {"output", "all"}:
        payload["output"] = MultimodalEmbeddingOutputConfig()
    if typed_sections == "all":
        payload["input"] = MultimodalEmbeddingInputConfig(episodes_lance_uri=uri)

    actual = ResolvedMultimodalEmbeddingConfig.model_validate(payload)

    assert actual == expected
    assert actual.output.staging_root_uri == str(tmp_path / "staging" / "multimodal-embedding")
    assert ResolvedMultimodalEmbeddingConfig.model_validate(actual.model_dump()) == actual
    assert actual.result_key_for_camera("front_wide") == "default_front_wide"


def test_null_duration_limit_is_rejected(tmp_path: Path) -> None:
    """Every run has a finite positive eligibility ceiling."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload["selection"] = {"max_episode_duration_s": None}
    with pytest.raises(ValidationError, match="max_episode_duration_s"):
        resolve_config_data(config_payload)


@pytest.mark.parametrize("video", [None, {}])
def test_omitted_camera_selection_defaults_to_all(video: dict[str, object] | None, tmp_path: Path) -> None:
    """Omitted or unset selectors resolve to the same explicit all-camera configuration."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    if video is None:
        del config_payload["video"]
    else:
        config_payload["video"] = video
    config = resolve_config_data(config_payload)
    assert config.video.camera_keys == "all"
    assert resolve_config_data(config.model_dump(mode="json")) == config


def test_empty_yaml_video_defaults_to_all(tmp_path: Path) -> None:
    """A bare YAML video key is null and selects all cameras just like an empty mapping."""
    config_payload = yaml.safe_load(
        "schema_version: 1\nkind: multimodal-embedding\ninput:\n"
        f"  episodes_lance_uri: {tmp_path / 'episodes.lance'}\nvideo:\n"
    )
    assert config_payload["video"] is None
    config = resolve_config_data(config_payload)
    assert config.video.camera_keys == "all"


@pytest.mark.parametrize("selection", [["rear", "front_wide"], ["front_wide"], "all"])
def test_camera_selection_supports_named_lists_and_all(selection: object, tmp_path: Path) -> None:
    """The multi-camera config resolves deterministically and round trips through canonical JSON."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload["video"] = {"camera_keys": selection}
    config = resolve_config_data(config_payload)
    assert config.result_key_for_camera("front_wide") == "default_front_wide"
    assert config.video.camera_keys == (sorted(selection) if isinstance(selection, list) else selection)
    assert resolve_config_data(config.model_dump(mode="json")) == config


@pytest.mark.parametrize(
    "video",
    [
        {"camera_key": "front_wide", "camera_keys": "all"},
        {"camera_keys": []},
        {"camera_keys": ["front_wide", "front_wide"]},
        {"camera_keys": [""]},
        {"camera_keys": [" front_wide"]},
        {"camera_keys": [1]},
        {"camera_keys": "front_wide"},
    ],
)
def test_camera_selection_rejects_ambiguous_or_invalid_requests(video: object, tmp_path: Path) -> None:
    """Explicit selectors must be unambiguous and contain valid camera names."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload["video"] = video
    with pytest.raises(ValidationError):
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
    assert template["video"] == {"camera_keys": "all"}
    assert template["output"]["tag_camera_key"] is True  # type: ignore[index]
    assert template["model"] == {"variant": "cosmos_embed1_336p"}
    assert template["selection"]["max_episode_duration_s"] == 60.0  # type: ignore[index]
    assert template["execution"]["inference_batch_size"] == 4  # type: ignore[index]
    assert template["execution"]["media_concurrency"] == template["execution"]["decode_concurrency"] == "auto"  # type: ignore[index]
    annotated = config_template_yaml()
    parsed = yaml.safe_load("\n".join(line for line in annotated.splitlines() if not line.startswith("#")))
    assert resolve_config_data(parsed).model_dump(mode="json") == resolve_config_data(template).model_dump(mode="json")
    required = {item["path"] for item in config_template_payload()["required_fields"]}
    assert "input.episodes_lance_uri" in required
    assert not {"video.camera_key", "video.camera_keys"} & required
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
                "  camera_keys: [front_wide]",
                "output:",
                "  result_key: default",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert MULTIMODAL_EMBEDDING_KIND.validate(config, ()) == {"ok": True}
    assert '"kind": "multimodal-embedding"' in MULTIMODAL_EMBEDDING_KIND.render(config, ())
    payload = MULTIMODAL_EMBEDDING_KIND.template_payload()
    assert payload["config"]["video"] == {"camera_keys": "all"}
    assert not any(item["path"].startswith("video.") for item in payload["required_fields"])


def test_single_camera_tagging_roundtrips_without_double_suffix(tmp_path: Path) -> None:
    """The config stores the base key even after rendering and dotted overrides."""
    config = resolve_config_data(
        _config_payload(str(tmp_path / "episodes.lance")),
        overrides=["output.result_key=review"],
    )
    assert config.output.result_key == "review"
    assert config.result_key_for_camera("front_wide") == "review_front_wide"
    reloaded = resolve_config_data(config.model_dump(mode="json"))
    assert reloaded == config
    assert reloaded.result_key_for_camera("front_wide") == "review_front_wide"


@pytest.mark.parametrize("camera", ["Front-Wide", "front-wide", "x" * 41])
def test_explicit_tagged_cameras_validate_the_effective_key(camera: str, tmp_path: Path) -> None:
    """Unsafe and overlong names fail during config resolution, before staging."""
    config_payload = _config_payload(str(tmp_path / "episodes.lance"))
    config_payload["video"] = {"camera_keys": [camera]}
    with pytest.raises(ValidationError, match="Invalid effective result key"):
        resolve_config_data(config_payload)


def test_multicamera_example_uses_caption_camera_tag_names() -> None:
    """The follow-up example is ready to pair with the pending caption examples."""
    root = Path(__file__).resolve().parents[5]
    config_payload = yaml.safe_load((root / "examples" / "multimodal-multicamera-embedding.yaml").read_text())
    config = resolve_config_data(config_payload)
    assert config.input.episodes_lance_uri == "/data/curated/full-rig/episodes.lance"
    assert config.video.camera_keys == ["front_wide", "left_fisheye"]
    assert config.output.result_key == "default"
    assert config.result_key_for_camera("front_wide") == "default_front_wide"
    assert config.result_key_for_camera("left_fisheye") == "default_left_fisheye"


def test_removed_scalar_option_is_rejected(tmp_path: Path) -> None:
    """The old scalar selector cannot silently activate a different execution path."""
    payload = _config_payload(str(tmp_path / "episodes.lance"))
    payload["video"] = {"camera_key": "front_wide"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        resolve_config_data(payload)


@pytest.mark.parametrize("camera", ["rear", "Front-Wide", "front-wide", "x" * 49])
def test_one_camera_can_explicitly_disable_tagging(camera: str, tmp_path: Path) -> None:
    """An explicit namespace follows captioning's opt-out without rewriting camera identity."""
    payload = _config_payload(str(tmp_path / "episodes.lance"))
    payload["video"] = {"camera_keys": [camera]}
    payload["output"] = MultimodalEmbeddingOutputConfig(result_key="review", tag_camera_key=False)
    config = resolve_config_data(payload)
    assert config.result_key_for_camera(camera) == "review"
    assert resolve_config_data(config.model_dump(mode="json")) == config


@pytest.mark.parametrize("selection", ["all", ["front_wide", "rear"]])
def test_tagging_cannot_be_disabled_for_multiple_or_discovered_cameras(selection: object, tmp_path: Path) -> None:
    """More than one camera must have independently named result fields."""
    payload = _config_payload(str(tmp_path / "episodes.lance"))
    payload["video"] = {"camera_keys": selection}
    payload["output"] = {"tag_camera_key": False}
    with pytest.raises(ValidationError, match="requires an explicit one-camera"):
        resolve_config_data(payload)


def test_nested_output_model_preserves_explicit_staging(tmp_path: Path) -> None:
    """Typed output objects retain an explicitly supplied staging location."""
    output = MultimodalEmbeddingOutputConfig(staging_root_uri=str(tmp_path / "custom-staging"))
    payload = _config_payload(str(tmp_path / "episodes.lance"))
    payload["output"] = output
    assert ResolvedMultimodalEmbeddingConfig.model_validate(payload).output == output
    assert output.model_fields_set == {"staging_root_uri"}
