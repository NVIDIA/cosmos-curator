# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public multimodal-caption configuration."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cosmos_curator.next.recipes.multimodal_caption.config import (
    config_schema_json,
    config_template,
    config_template_payload,
    config_template_yaml,
    resolve_config_data,
    resolved_config_to_json,
)
from cosmos_curator.next.recipes.multimodal_caption.config import (
    resolve_config as resolve_caption_config,
)
from cosmos_curator.next.recipes.multimodal_split.config import resolve_config as resolve_split_config

_OMITTED = object()


def _raw(uri: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "multimodal-caption",
        "input": {"episodes_lance_uri": uri},
        "context": {"variant": "video_only", "camera_key": "front_wide"},
        "inference": {"prompt": {"mode": "default"}},
        "output": {"result_key": "default"},
    }


def test_required_selections_and_defaults_are_resolved(tmp_path: Path) -> None:
    """Required selections survive resolution while model and output identity default."""
    config = resolve_config_data(_raw(str(tmp_path / "episodes.lance")))

    assert config.context.camera_key == "front_wide"
    assert config.context.variant == "video_only"
    assert config.model.variant == "qwen3_8_27b_fp8"
    assert config.inference.prompt.mode == "default"
    assert config.output.result_key == "default"
    assert config.output.staging_root_uri == str(tmp_path / "staging" / "multimodal-caption")


@pytest.mark.parametrize("missing", ["context", "inference"])
def test_context_and_prompt_are_required(missing: str, tmp_path: Path) -> None:
    """Every invocation explicitly selects context and prompt mode."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw.pop(missing)
    with pytest.raises(ValidationError, match="Field required"):
        resolve_config_data(raw)


def test_omitted_empty_and_explicit_default_output_resolve_identically(tmp_path: Path) -> None:
    """All default-output spellings produce one canonical output and workspace."""
    uri = str(tmp_path / "episodes.lance")
    omitted = _raw(uri)
    omitted.pop("output")
    empty = _raw(uri)
    empty["output"] = {}
    explicit = _raw(uri)

    resolved = [resolve_config_data(raw).output for raw in (omitted, empty, explicit)]

    assert resolved[0] == resolved[1] == resolved[2]
    assert resolved[0].result_key == "default"
    assert resolved[0].staging_root_uri == str(tmp_path / "staging" / "multimodal-caption")


def test_explicit_null_output_is_invalid(tmp_path: Path) -> None:
    """An optional output block may be omitted but is never nullable."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw["output"] = None
    with pytest.raises(ValidationError):
        resolve_config_data(raw)


@pytest.mark.parametrize(
    "change",
    [
        {"context": {"variant": "video_only"}},
        {"context": {"variant": "video_only", "camera_key": ""}},
        {"context": {"variant": "video_only_av", "camera_key": "front_wide"}},
        {"unknown": True},
    ],
)
def test_required_camera_variant_and_strict_shape(change: dict[str, object], tmp_path: Path) -> None:
    """The public model rejects missing, retired, and unknown input."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw.update(change)
    with pytest.raises(ValidationError):
        resolve_config_data(raw)


def test_video_only_forbids_ego_config(tmp_path: Path) -> None:
    """The context-free mode cannot accept dormant Ego settings."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw["context"] = {
        "variant": "video_only",
        "camera_key": "front_wide",
        "ego": {"sensor_key": "ego", "topic": "/ego", "frame": "world"},
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        resolve_config_data(raw)


@pytest.mark.parametrize("variant", ["ego_trajectory_summary", "ego_trajectory_absolute_poses"])
def test_ego_variants_require_exact_source_config(variant: str, tmp_path: Path) -> None:
    """Both Ego contexts expose only sensor key, topic, and frame."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw["context"] = {"variant": variant, "camera_key": "front_wide"}
    with pytest.raises(ValidationError, match="Field required"):
        resolve_config_data(raw)

    raw["context"] = {
        "variant": variant,
        "camera_key": "front_wide",
        "ego": {"sensor_key": "ego", "topic": "/ego", "frame": "world"},
    }
    assert resolve_config_data(raw).context.ego is not None  # type: ignore[union-attr]

    for forbidden in ("schema_name", "protobuf_mapping", "cadence_period_ns", "max_delta_ns"):
        invalid = dict(raw)
        invalid["context"] = dict(raw["context"])  # type: ignore[arg-type]
        invalid["context"]["ego"] = dict(raw["context"]["ego"], **{forbidden: "override"})  # type: ignore[index]
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            resolve_config_data(invalid)


@pytest.mark.parametrize(
    ("prompt", "result_key", "expected"),
    [
        ({"mode": "default"}, _OMITTED, "default"),
        ({"mode": "default"}, "default", "default"),
        ({"mode": "default"}, "front_wide", "front_wide"),
        ({"mode": "default"}, "default_front_wide", "default_front_wide"),
        ({"mode": "default", "template": "forbidden"}, "default", None),
        ({"mode": "custom", "template": "exact prompt"}, _OMITTED, None),
        ({"mode": "custom", "template": "exact prompt"}, "default", None),
        ({"mode": "custom", "template": "exact prompt"}, "driving_summary_v2", "driving_summary_v2"),
        ({"mode": "custom", "template": "exact prompt"}, "default_custom", None),
        ({"mode": "custom", "template": "exact prompt"}, "defaulted", None),
        ({"mode": "custom", "template": "  \n"}, "custom", None),
    ],
)
def test_prompt_mode_and_result_namespace_are_strict(
    prompt: dict[str, str], result_key: object, expected: str | None, tmp_path: Path
) -> None:
    """Custom prompts cannot claim any base key beginning with default."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw["inference"] = {"prompt": prompt}
    if result_key is _OMITTED:
        raw.pop("output")
    else:
        raw["output"] = {"result_key": result_key}
    if expected is not None:
        config = resolve_config_data(raw)
        assert config.output.result_key == expected
    else:
        with pytest.raises(ValidationError):
            resolve_config_data(raw)


@pytest.mark.parametrize("result_key", ["Default", "9result", "has-dash", "a" * 49, None, 7])
def test_result_key_formula_segment_is_validated(result_key: object, tmp_path: Path) -> None:
    """Result keys are safe deterministic field-name segments."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw["output"] = {"result_key": result_key}
    with pytest.raises(ValidationError):
        resolve_config_data(raw)


def test_custom_prompt_bytes_are_preserved(tmp_path: Path) -> None:
    """Configuration parsing does not trim or normalize a custom prompt."""
    template = "  exact {braces}\nsecond line  "
    raw = _raw(str(tmp_path / "episodes.lance"))
    raw["inference"] = {"prompt": {"mode": "custom", "template": template}}
    raw["output"] = {"result_key": "custom_prompt"}
    config = resolve_config_data(raw)
    assert config.inference.prompt.template == template  # type: ignore[union-attr]


def test_generated_configuration_surfaces_are_consistent() -> None:
    """Template, annotated YAML, payload, and schema expose the same required surface."""
    template = config_template()
    assert template["context"] == {"variant": "video_only", "camera_key": "front_wide"}
    assert template["model"] == {"variant": "qwen3_8_27b_fp8"}
    assert template["inference"] == {"prompt": {"mode": "default"}}
    assert template["output"]["result_key"] == "default"  # type: ignore[index]
    annotated = config_template_yaml()
    assert "Custom mode requires an explicit base result_key that does not start with default." in annotated
    assert "Custom Ego prompts require exactly one literal {ego_trajectory_context} sentinel." in annotated
    parsed = yaml.safe_load("\n".join(line for line in annotated.splitlines() if not line.startswith("#")))
    assert resolve_config_data(parsed).model_dump(mode="json") == resolve_config_data(template).model_dump(mode="json")
    required = {item["path"] for item in config_template_payload()["required_fields"]}
    assert {"context.variant", "inference.prompt.mode"} <= required
    assert "output.result_key" not in required
    schema = json.loads(config_schema_json())
    assert "output" not in schema["required"]
    output_schema = schema["$defs"]["MultimodalCaptionOutputConfig"]
    assert "result_key" not in output_schema.get("required", [])
    assert output_schema["properties"]["result_key"]["default"] == "default"
    assert output_schema["properties"]["tag_camera_key"]["default"] is False
    assert template["output"]["tag_camera_key"] is False
    assert "ego_trajectory_absolute_poses" in json.dumps(schema)
    assert "qwen3_8_27b_fp8" in json.dumps(schema)


@pytest.mark.parametrize("tag", [False, True])
@pytest.mark.parametrize("base", [_OMITTED, "default", "experiment_02", "front_wide"])
def test_camera_tagging_preserves_reloadable_base_key(base: object, tmp_path: Path, *, tag: bool) -> None:
    """Tagging appends exactly once, including to a camera-named base."""
    output: dict[str, object] = {"tag_camera_key": tag}
    if base is not _OMITTED:
        output["result_key"] = base
    config = resolve_config_data(_raw(str(tmp_path / "episodes.lance")) | {"output": output})
    expected_base = "default" if base is _OMITTED else base
    assert config.output.result_key == expected_base
    assert config.effective_result_key == (f"{expected_base}_front_wide" if tag else expected_base)
    reloaded = resolve_config_data(json.loads(resolved_config_to_json(config)))
    assert reloaded == config
    assert reloaded.effective_result_key == config.effective_result_key


def test_tagging_resolves_after_cli_overrides(tmp_path: Path) -> None:
    """Overrides choose the camera and base before validating the effective key."""
    config = resolve_config_data(
        _raw(str(tmp_path / "episodes.lance")),
        overrides=["output.tag_camera_key=true", "output.result_key=run2", "context.camera_key=rear_fisheye"],
    )
    assert config.effective_result_key == "run2_rear_fisheye"


@pytest.mark.parametrize("camera", ["Front", "front.wide", "front-wide", "x" * 41])
def test_tagging_rejects_invalid_effective_keys_with_guidance(camera: str, tmp_path: Path) -> None:
    """Wider camera alphabets remain usable through explicit untagged keys."""
    raw = _raw(str(tmp_path / "episodes.lance")) | {
        "context": {"variant": "video_only", "camera_key": camera},
        "output": {"tag_camera_key": True},
    }
    with pytest.raises(ValidationError, match="Turn tag_camera_key off"):
        resolve_config_data(raw)
    assert resolve_config_data(raw | {"output": {"result_key": "camera_run"}}).effective_result_key == "camera_run"


def test_effective_key_length_boundary_and_staging_only_default(tmp_path: Path) -> None:
    """The composed key accepts 48 characters; staging-only output keeps default naming."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    config = resolve_config_data(raw | {"output": {"result_key": "a" * 37, "tag_camera_key": True}})
    assert len(config.effective_result_key) == 48
    with pytest.raises(ValidationError, match="48 characters"):
        resolve_config_data(raw | {"output": {"result_key": "a" * 38, "tag_camera_key": True}})
    config = resolve_config_data(raw | {"output": {"staging_root_uri": str(tmp_path / "stage")}})
    assert config.effective_result_key == "default"


@pytest.mark.parametrize("tag", [False, True])
@pytest.mark.parametrize("base", [_OMITTED, "default", "default_front_wide", "defaulted"])
def test_custom_prefix_is_rejected_before_tagging(base: object, tmp_path: Path, *, tag: bool) -> None:
    """Tagging does not make a reserved custom base acceptable."""
    output: dict[str, object] = {"tag_camera_key": tag}
    if base is not _OMITTED:
        output["result_key"] = base
    raw = _raw(str(tmp_path / "episodes.lance")) | {
        "output": output,
        "inference": {"prompt": {"mode": "custom", "template": "Describe."}},
    }
    with pytest.raises(ValidationError, match="does not start with 'default'"):
        resolve_config_data(raw)
    config = resolve_config_data(raw | {"output": {"result_key": "custom", "tag_camera_key": tag}})
    assert config.effective_result_key == ("custom_front_wide" if tag else "custom")


def test_explicit_null_tag_is_rejected(tmp_path: Path) -> None:
    """The tagging flag is a strict non-null boolean."""
    with pytest.raises(ValidationError):
        resolve_config_data(_raw(str(tmp_path / "episodes.lance")) | {"output": {"tag_camera_key": None}})


def test_all_three_models_are_supported(tmp_path: Path) -> None:
    """The recipe accepts its exact model set and rejects unknown keys."""
    raw = _raw(str(tmp_path / "episodes.lance"))
    assert {
        resolve_config_data(raw | {"model": {"variant": variant}}).model.variant
        for variant in ("qwen3_8_27b_fp8", "qwen3_8_27b", "cosmos3_nano")
    } == {"qwen3_8_27b_fp8", "qwen3_8_27b", "cosmos3_nano"}
    with pytest.raises(ValidationError):
        resolve_config_data(raw | {"model": {"variant": "future_model"}})


def test_multicamera_examples_resolve() -> None:
    """Public examples use their recipe resolvers; runtime images may omit examples."""
    examples = Path(__file__).resolve().parents[5] / "examples"
    if not examples.exists():
        pytest.skip("examples directory is not included in runtime images")

    resolve_split_config(examples / "multimodal-multicamera-split.yaml")
    for selection in ("front", "side", "rear-ego"):
        resolve_caption_config(examples / f"multimodal-multicamera-{selection}-caption.yaml")
