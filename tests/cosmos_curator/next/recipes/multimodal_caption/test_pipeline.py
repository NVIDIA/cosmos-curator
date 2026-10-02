# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused startup tests for compositional caption selection and preflight."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.multimodal_caption import pipeline
from cosmos_curator.next.recipes.multimodal_caption.config import resolve_config_data
from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    CONTRACT_DIGEST_METADATA,
    EGO_DECODER_MAPPING_DIGEST,
    CaptionInputProfile,
    caption_contract_digest,
    normalized_caption_contract,
    resolve_model,
    result_field_names,
    result_fields,
)
from cosmos_curator.next.recipes.multimodal_split.records import EPISODE_SCHEMA
from cosmos_curator.next.recipes.video_caption.bindings import CaptionPublicationBindings
from cosmos_curator.next.recipes.video_caption.inference import caption_llm_preprocess
from cosmos_curator.next.utils.identity import canonical_digest


def _config(
    tmp_path: Path,
    variant: str,
    *,
    prompt: dict[str, str] | None = None,
    result_key: str = "default",
) -> object:
    context: dict[str, object] = {"variant": variant, "camera_key": "front_wide"}
    if variant != "video_only":
        context["ego"] = {"sensor_key": "ego", "topic": "/ego", "frame": "world"}
    return resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-caption",
            "input": {"episodes_lance_uri": str(tmp_path / "episodes.lance")},
            "context": context,
            "inference": {"prompt": prompt or {"mode": "default"}},
            "output": {"result_key": result_key},
        }
    )


def test_video_only_does_not_load_ego_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Context-free startup never touches the Ego decoder resource."""
    monkeypatch.setattr(
        pipeline,
        "_load_ego_mapping",
        lambda: pytest.fail("video_only must not load the Ego mapping"),
    )

    class ReachedModelResolutionError(Exception):
        pass

    monkeypatch.setattr(
        pipeline,
        "resolve_model",
        lambda _variant: (_ for _ in ()).throw(ReachedModelResolutionError),
    )
    with pytest.raises(ReachedModelResolutionError):
        pipeline.run_config(_config(tmp_path, "video_only"))  # type: ignore[arg-type]


def test_recipe_owned_mapping_shape_and_digest_are_pinned() -> None:
    """Production loads the exact parsed mapping with explicit alignment target."""
    mapping = pipeline._load_ego_mapping()
    assert mapping["fields"]["sensor_timestamp_ns"] == {  # type: ignore[index]
        "from": "timestamp",
        "type": "protobuf_timestamp",
    }
    assert mapping["fields"]["align_timestamp_ns"] == {  # type: ignore[index]
        "from": "$mcap.logtime",
        "type": "timestamp",
        "unit": "ns",
    }
    assert mapping["fields"]["pose_valid"] == {"default": True}  # type: ignore[index]
    assert canonical_digest(mapping) == EGO_DECODER_MAPPING_DIGEST


@pytest.mark.parametrize("variant", ["ego_trajectory_summary", "ego_trajectory_absolute_poses"])
def test_resolved_prompt_is_bound_into_ego_adapter(variant: str, tmp_path: Path) -> None:
    """The adapter receives the single already resolved run template."""
    custom = "keep {unrelated} exactly: {ego_trajectory_context}"
    config = _config(tmp_path, variant, prompt={"mode": "custom", "template": custom}, result_key="custom")
    definition, ego, resolved = pipeline._resolve_selected_context(config)  # type: ignore[arg-type]
    assert definition.input_profile is CaptionInputProfile.EGO_TRAJECTORY
    _, context_stage, carried = pipeline._resolve_input_runtime(
        definition,
        ego,
        camera_key="front_wide",
        prompt_template=resolved,
        mapping=pipeline._load_ego_mapping(),
    )
    assert isinstance(context_stage, partial)
    assert context_stage.keywords["prompt_template"] == custom
    assert context_stage.keywords["prompt_template"] != definition.default_prompt_template
    assert carried == ("context_digest",)


def test_camera_preflight_failure_leaves_result_fields_absent(
    episode_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
) -> None:
    """Camera selection is validated before digest-tagged namespace registration."""
    uri, _ = episode_dataset_factory()
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-caption",
            "input": {"episodes_lance_uri": uri},
            "context": {"variant": "video_only", "camera_key": "missing"},
            "inference": {"prompt": {"mode": "default"}},
            "output": {"result_key": "default"},
        }
    )
    with pytest.raises(ValueError, match="0 camera descriptors"):
        pipeline.run_config(config)
    expected = result_field_names("video_only", "default", "qwen3_8_27b_fp8")
    assert set(expected).isdisjoint(lance.dataset(uri).schema.names)


def test_ego_sensor_preflight_failure_leaves_result_fields_absent(
    episode_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    episode_factory: Callable[..., dict[str, object]],
    sensor_factory: Callable[..., dict[str, object]],
) -> None:
    """Non-sensor Ego rows fail before the result namespace is claimed."""
    row = episode_factory(
        0,
        sensor_media=[sensor_factory()],
        timestamp_source=TimestampSource.CONTAINER_PTS,
    )
    uri, _ = episode_dataset_factory(rows=[row])
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-caption",
            "input": {"episodes_lance_uri": uri},
            "context": {
                "variant": "ego_trajectory_summary",
                "camera_key": "front_wide",
                "ego": {"sensor_key": "ego", "topic": "/ego", "frame": "world"},
            },
            "inference": {"prompt": {"mode": "default"}},
            "output": {"result_key": "default"},
        }
    )
    with pytest.raises(ValueError, match="Ego caption contexts require 'sensor'"):
        pipeline.run_config(config)
    expected = result_field_names("ego_trajectory_summary", "default", "qwen3_8_27b_fp8")
    assert set(expected).isdisjoint(lance.dataset(uri).schema.names)


def test_ego_invalid_span_preflight_leaves_result_fields_absent(
    episode_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    episode_factory: Callable[..., dict[str, object]],
    sensor_factory: Callable[..., dict[str, object]],
) -> None:
    """A non-positive Ego span fails before the result namespace is claimed."""
    row = episode_factory(0, sensor_media=[sensor_factory()])
    row["exclusive_end_ns"] = row["start_ns"]
    uri, _ = episode_dataset_factory(rows=[row])
    config = resolve_config_data(
        {
            "schema_version": 1,
            "kind": "multimodal-caption",
            "input": {"episodes_lance_uri": uri},
            "context": {
                "variant": "ego_trajectory_summary",
                "camera_key": "front_wide",
                "ego": {"sensor_key": "ego", "topic": "/ego", "frame": "world"},
            },
            "inference": {"prompt": {"mode": "default"}},
            "output": {"result_key": "default"},
        }
    )
    with pytest.raises(ValueError, match="must have a positive duration"):
        pipeline.run_config(config)
    expected = result_field_names("ego_trajectory_summary", "default", "qwen3_8_27b_fp8")
    assert set(expected).isdisjoint(lance.dataset(uri).schema.names)


def test_video_only_default_and_custom_prompts_bind_exactly(tmp_path: Path) -> None:
    """Context-free bindings preserve exact prompts, identity, and schema metadata."""
    custom = "Describe this clip while preserving {anything} literally."
    cases = (
        (_config(tmp_path, "video_only"), "default"),
        (_config(tmp_path, "video_only", prompt={"mode": "custom", "template": custom}, result_key="custom"), "custom"),
    )
    identities: list[tuple[tuple[str, str], dict[str, object], str]] = []
    for raw_config, result_key in cases:
        definition, ego, prompt = pipeline._resolve_selected_context(raw_config)  # type: ignore[arg-type]
        input_bindings, context_stage, carried = pipeline._resolve_input_runtime(
            definition,
            ego,
            camera_key="front_wide",
            prompt_template=prompt,
            mapping=None,
        )
        assert context_stage is None
        assert carried == ()
        assert input_bindings.staged_binding_columns
        model = resolve_model("qwen3_8_27b_fp8")
        contract = normalized_caption_contract(model, "front_wide", definition, ego, prompt, result_key)
        digest = caption_contract_digest(model, "front_wide", definition, ego, prompt, result_key)
        fields = result_fields(definition.variant, result_key, model.key, ego=ego, digest=digest)
        assert all(field.metadata[CONTRACT_DIGEST_METADATA] == digest.encode() for field in fields)
        row = {"video_data_url": "data:video/mp4;base64,AAAA", "download_error_type": None}
        request = caption_llm_preprocess(
            row,
            prompt_template=prompt,
            request_mm_processor_kwargs={},
            sampling_params={"max_tokens": 1},
        )
        assert request["messages"][0]["content"][1] == {"type": "text", "text": prompt}
        assert "rendered_prompt" not in row
        identities.append((result_field_names(definition.variant, result_key, model.key), contract, digest))

    assert identities[0][0] != identities[1][0]
    assert identities[0][1] != identities[1][1]
    assert identities[0][2] != identities[1][2]
    assert "{anything}" in custom


def test_changed_prompt_cannot_reuse_null_registered_namespace(
    episode_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
) -> None:
    """Digest-tagged fields reject a changed contract even before any terminal value exists."""
    uri, _ = episode_dataset_factory()

    def config(prompt: str) -> object:
        return resolve_config_data(
            {
                "schema_version": 1,
                "kind": "multimodal-caption",
                "input": {"episodes_lance_uri": uri},
                "context": {"variant": "video_only", "camera_key": "front_wide"},
                "inference": {"prompt": {"mode": "custom", "template": prompt}},
                "output": {"result_key": "custom"},
            }
        )

    with pytest.raises(FileNotFoundError, match="Model weights"):
        pipeline.run_config(config("first prompt"))  # type: ignore[arg-type]
    fields = result_field_names("video_only", "custom", "qwen3_8_27b_fp8")
    assert set(fields) <= set(lance.dataset(uri).schema.names)
    before = lance.dataset(uri).to_table().to_pylist()
    with pytest.raises(ValueError, match=r"incompatible.*different contract.*camera, prompt, model revision") as caught:
        pipeline.run_config(config("changed prompt"))  # type: ignore[arg-type]
    assert "set tag_camera_key: true" in str(caught.value)
    assert lance.dataset(uri).to_table().to_pylist() == before


def test_ambiguous_tagged_keys_reject_incompatible_camera(
    episode_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    episode_factory: Callable[..., dict[str, object]],
    camera_factory: Callable[..., dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concatenations producing a_b_c cannot share a different camera contract."""
    uri, _ = episode_dataset_factory(
        rows=[
            episode_factory(
                0,
                camera_media=[
                    camera_factory("c", "1"),
                    camera_factory("b_c", "2"),
                ],
            )
        ]
    )
    raw = {
        "schema_version": 1,
        "kind": "multimodal-caption",
        "input": {"episodes_lance_uri": uri},
        "context": {"variant": "video_only", "camera_key": "c"},
        "inference": {"prompt": {"mode": "default"}},
        "output": {"result_key": "a_b", "tag_camera_key": True},
    }

    def stop_before_inference(*_args: object, **_kwargs: object) -> None:
        raise InterruptedError

    monkeypatch.setattr(pipeline, "ensure_workspace", stop_before_inference)
    with pytest.raises(InterruptedError):
        pipeline.run_config(resolve_config_data(raw))
    version = lance.dataset(uri).version
    other = resolve_config_data(
        raw
        | {
            "context": {"variant": "video_only", "camera_key": "b_c"},
            "output": {"result_key": "a", "tag_camera_key": True},
        }
    )
    with pytest.raises(ValueError, match="Choose a new base result_key") as caught:
        pipeline.run_config(other)
    assert "set tag_camera_key: true" not in str(caught.value)
    assert lance.dataset(uri).version == version


@pytest.mark.parametrize("failure", ["partial", "base_schema", "unrelated", "reopen"])
def test_registration_errors_without_established_collision_keep_diagnostic(
    failure: str,
    tmp_path: Path,
    episode_factory: Callable[..., dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither partial fields nor invalid base schemas justify a collision hint."""
    config = _config(tmp_path, "video_only")
    caption_name, _ = result_field_names("video_only", "default", "qwen3_8_27b_fp8")
    schema = EPISODE_SCHEMA
    if failure == "partial":
        schema = schema.append(pa.field(caption_name, pa.string()))
    elif failure == "base_schema":
        schema = schema.remove(schema.get_field_index("source_id"))
        for name in result_field_names("video_only", "default", "qwen3_8_27b_fp8"):
            schema = schema.append(pa.field(name, pa.string()))
    lance.write_dataset(
        pa.Table.from_pylist([episode_factory(0)], schema=schema),
        str(tmp_path / "episodes.lance"),
        data_storage_version="2.2",
    )
    monkeypatch.setattr(pipeline, "validate_episode_input", lambda *_a, **_k: None)
    original = ValueError(f"original {failure} diagnostic")

    def reject(*_args: object, **_kwargs: object) -> None:
        if failure == "reopen":

            def unavailable(*_a: object, **_k: object) -> None:
                msg = "table temporarily unavailable"
                raise OSError(msg)

            monkeypatch.setattr(pipeline.lance, "dataset", unavailable)
        raise original

    monkeypatch.setattr(pipeline, "ensure_caption_fields", reject)
    with pytest.raises(ValueError, match=f"original {failure} diagnostic") as caught:
        pipeline.run_config(config)  # type: ignore[arg-type]
    assert caught.value is original
    assert str(caught.value) == f"original {failure} diagnostic"


def test_combined_ego_preflight_precedes_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ego startup scans once, registers, and then reopens the table."""
    config = SimpleNamespace(
        input=SimpleNamespace(episodes_lance_uri="/episodes.lance"),
        context=SimpleNamespace(camera_key="front_wide"),
        execution=SimpleNamespace(commit_attempts=5),
    )
    definition, ego, _ = pipeline._resolve_selected_context(
        resolve_config_data(
            {
                "schema_version": 1,
                "kind": "multimodal-caption",
                "input": {"episodes_lance_uri": "/episodes.lance"},
                "context": {
                    "variant": "ego_trajectory_summary",
                    "camera_key": "front_wide",
                    "ego": {"sensor_key": "ego", "topic": "/ego", "frame": "world"},
                },
                "inference": {"prompt": {"mode": "default"}},
                "output": {"result_key": "default"},
            }
        )
    )
    before = SimpleNamespace(name="before")
    after = SimpleNamespace(name="after")
    datasets = iter((before, after))
    events: list[str] = []
    monkeypatch.setattr(pipeline.lance, "dataset", lambda *_args, **_kwargs: next(datasets))
    monkeypatch.setattr(
        pipeline,
        "validate_ego_episode_input",
        lambda *_args, **_kwargs: events.append("validate") or (None, None),
    )
    monkeypatch.setattr(
        pipeline,
        "ensure_caption_fields",
        lambda *_args, **_kwargs: events.append("register"),
    )
    dataset, _, _ = pipeline._preflight_and_register(config, definition, ego, SimpleNamespace(), None)  # type: ignore[arg-type]
    assert dataset is after
    assert events == ["validate", "register"]


def test_workspace_and_publication_identity_are_complete(tmp_path: Path) -> None:
    """Recovery and transaction surfaces carry direct identifiers plus the contract digest."""
    config = _config(tmp_path, "ego_trajectory_summary")
    definition, ego, prompt = pipeline._resolve_selected_context(config)  # type: ignore[arg-type]
    model = resolve_model("qwen3_8_27b_fp8")
    contract = normalized_caption_contract(model, "front_wide", definition, ego, prompt, "default")
    digest = caption_contract_digest(model, "front_wide", definition, ego, prompt, "default")
    caption_field, metadata_field = result_field_names(definition.variant, "default", model.key)
    manifest = pipeline._workspace_manifest(
        uri="/episodes.lance",
        caption_field=caption_field,
        metadata_field=metadata_field,
        contract=contract,
        definition=definition,
        result_key="default",
        camera_key="front_wide",
        model=model,
        digest=digest,
    )
    assert manifest["caption_contract"] == contract
    assert manifest["model_key"] == model.key
    assert manifest["context_variant"] == definition.variant
    assert manifest["result_key"] == "default"
    assert manifest["caption_contract_digest"] == digest
    assert "prompt_mode" not in manifest


def test_run_composes_transaction_identity_and_context_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selected context and result identity reach bindings without a pair table."""

    class BindingsCapturedError(Exception):
        pass

    captured: dict[str, object] = {}

    def capture_bindings(**kwargs: object) -> None:
        captured.update(kwargs)
        raise BindingsCapturedError

    monkeypatch.setattr(pipeline, "CaptionRunBindings", capture_bindings)
    with pytest.raises(BindingsCapturedError):
        pipeline.run_config(_config(tmp_path, "ego_trajectory_absolute_poses"))  # type: ignore[arg-type]

    publication = captured["publication"]
    assert isinstance(publication, CaptionPublicationBindings)
    properties = dict(publication.transaction_properties)
    assert properties["kind"] == "multimodal-caption"
    assert properties["result_key"] == "default"
    assert properties["context_variant"] == "ego_trajectory_absolute_poses"
    assert properties["model_key"] == "qwen3_8_27b_fp8"
    assert len(properties["caption_contract_digest"]) == 64
    assert "prompt_mode" not in properties
    context_stage = captured["context_stage"]
    assert isinstance(context_stage, partial)
    assert context_stage.keywords["build_context"].__name__ == "build_ego_trajectory_absolute_poses"
