# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the versioned episode ``video_only`` result contract."""

from typing import get_args

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.multimodal_caption.config import ModelVariant
from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    build_terminal_result_row,
    caption_contract_digest,
    caption_inference_bindings,
    normalized_caption_contract,
    resolve_model,
    result_field_names,
    result_fields,
    staged_result_schema,
    validate_terminal_row,
)
from cosmos_curator.next.recipes.video_caption.contracts import (
    normalized_caption_contract as normalized_clip_caption_contract,
)
from cosmos_curator.next.recipes.video_caption.contracts import (
    resolve_model_spec,
    resolved_max_output_tokens,
    resolved_sampling_params,
)
from cosmos_curator.next.recipes.video_caption.inference import caption_llm_preprocess


@pytest.mark.parametrize(
    ("model_key", "caption_name"),
    [
        ("qwen3_8_27b_fp8", "caption_video_only_qwen3_8_27b_fp8_v1"),
        ("qwen3_8_27b", "caption_video_only_qwen3_8_27b_v1"),
        ("cosmos3_nano", "caption_video_only_cosmos3_nano_v1"),
    ],
)
def test_public_fields_and_metadata_are_exact(model_key: str, caption_name: str) -> None:
    """Both model variants publish their exact versioned field sets."""
    fields = result_fields(model_key)
    metadata = fields.field(1).type

    assert fields.names == [caption_name, f"{caption_name}_metadata"]
    assert all(field.nullable for field in fields)
    assert fields.field(0).type == pa.large_string()
    assert [field.name for field in metadata] == [
        "caption_schema_version",
        "result_variant",
        "contract_digest",
        "episode_id",
        "materialization_digest",
        "camera_key",
        "media_sha256",
        "status",
        "model_id",
        "model_revision",
        "prompt",
        "prompt_token_count",
        "generated_token_count",
        "error_type",
        "error_message",
    ]
    assert fields.field(0).metadata == {
        b"cosmos_curator.owner": b"multimodal-caption",
        b"cosmos_curator.field_set": caption_name.encode(),
    }


def test_every_accepted_config_variant_has_one_complete_contract() -> None:
    """Each variant the config accepts resolves pinned weights, owned fields, and a runtime."""
    variants = get_args(ModelVariant)
    caption_names = set()

    for variant in variants:
        caption_name, metadata_name = result_field_names(variant)
        contract = normalized_caption_contract(resolve_model(variant), "front_wide")

        assert metadata_name == f"{caption_name}_metadata"
        assert [field["name"] for field in contract["fields"]] == [caption_name, metadata_name]
        caption_names.add(caption_name)

    assert len(caption_names) == len(variants)
    with pytest.raises(KeyError, match="no variant binding for 'qwen3_8_27b_awq'"):
        caption_inference_bindings("qwen3_8_27b_awq")


@pytest.mark.parametrize(
    (
        "model_key",
        "expected_digest",
        "expected_fps",
        "expected_request_mm_processor_kwargs",
        "expected_do_sample_frames",
    ),
    [
        (
            "qwen3_8_27b_fp8",
            "0fa3f137b0b415ea49b338bd1eafa6af293566ead73340f985fdd82f0dbef2d4",
            2,
            None,
            False,
        ),
        (
            "qwen3_8_27b",
            "e4c306c234aa8b7cc255b0c99da02a559936ca9ca5eed49995ee7267c94b1c43",
            2,
            None,
            False,
        ),
        (
            "cosmos3_nano",
            "deab31e64c03b92ba427b4df5b1b4377d9201e472f318e59649cd83d5042ac23",
            4,
            {},
            None,
        ),
    ],
)
def test_contract_digest_is_pinned_and_binds_camera(
    model_key: str,
    expected_digest: str,
    expected_fps: int,
    expected_request_mm_processor_kwargs: dict[str, object] | None,
    expected_do_sample_frames: object,
) -> None:
    """Each model/camera contract has a pinned digest and changes with the camera."""
    model = resolve_model(model_key)
    digest = caption_contract_digest(model, "front_wide")
    other_camera_digest = caption_contract_digest(model, "front_left")

    assert digest == expected_digest
    assert digest != other_camera_digest
    contract = normalized_caption_contract(model, "front_wide")
    assert contract["episode_contract"]["camera_key"] == "front_wide"
    expected_video = {
        "container": "exact_episode_camera_mp4",
        "num_frames": -1,
        "fps": expected_fps,
        "do_sample_frames": expected_do_sample_frames,
        "longest_edge_override": None,
        "audio": "ignored",
    }
    if expected_request_mm_processor_kwargs is not None:
        expected_video["request_mm_processor_kwargs"] = expected_request_mm_processor_kwargs
    assert contract["video"] == expected_video


def test_shared_sampling_matches_runtime_and_both_contracts() -> None:
    """The shared request and both result digests use one sampling contract."""
    qwen_runtime = caption_inference_bindings("qwen3_8_27b_fp8")
    request = caption_llm_preprocess(
        {"video_data_url": "data:video/mp4;base64,AAAA", "download_error_type": None},
        request_mm_processor_kwargs=qwen_runtime.request_mm_processor_kwargs,
        sampling_params=resolved_sampling_params(qwen_runtime),
    )
    clip_contract = normalized_clip_caption_contract(resolve_model_spec("qwen3_8_27b_fp8"))
    episode_contract = normalized_caption_contract(resolve_model("qwen3_8_27b_fp8"), "front_wide")

    assert clip_contract["sampling"] == episode_contract["sampling"] == request["sampling_params"]


def test_cosmos3_contract_binds_native_runtime_and_closed_thinking() -> None:
    """Cosmos3's result identity records its native vLLM launch and prompt form."""
    model = resolve_model("cosmos3_nano")
    runtime = caption_inference_bindings(model.key)
    request = caption_llm_preprocess(
        {"video_data_url": "data:video/mp4;base64,AAAA", "download_error_type": None},
        assistant_prefill=runtime.assistant_prefill,
        request_mm_processor_kwargs=runtime.request_mm_processor_kwargs,
        sampling_params=resolved_sampling_params(runtime),
    )
    contract = normalized_caption_contract(model, "front_wide")

    assert request["messages"][-1] == {"role": "assistant", "content": "<think></think>"}
    assert "mm_processor_kwargs" not in request
    assert contract["message"][-1] == {"role": "assistant", "content": "<think></think>"}
    assert contract["chat_template"] == {"source": "model", "kwargs": {}}
    assert runtime.sampling_params is not None
    assert contract["sampling"] == request["sampling_params"] == dict(runtime.sampling_params)
    assert contract["terminal_states"]["truncated"]["generated_token_count_if_present"] == {
        "greater_than_or_equal": resolved_max_output_tokens(runtime)
    }
    assert contract["model_runtime"] == {
        "prepare_model_config_kwargs": {
            "hf_overrides": {"architectures": ["Cosmos3ForConditionalGeneration"]},
        },
        "engine_kwargs": {
            "hf_overrides": {"architectures": ["Cosmos3ForConditionalGeneration"]},
            "mm_encoder_tp_mode": "data",
            "allowed_local_media_path": "/",
            "limit_mm_per_prompt": {"images": 0, "video": 1},
            "max_model_len": 32_768,
        },
    }
    assert contract["video"]["fps"] == 4
    assert contract["video"]["request_mm_processor_kwargs"] == {}
    assert contract["video"]["do_sample_frames"] is None


def test_contract_runtime_mappings_do_not_alias_bindings_or_each_other() -> None:
    """A caller cannot mutate static Cosmos3 bindings through a returned contract."""
    contract = normalized_caption_contract(resolve_model("cosmos3_nano"), "front_wide")
    runtime = contract["model_runtime"]
    runtime["prepare_model_config_kwargs"]["hf_overrides"]["architectures"].append("MutatedArchitecture")

    assert runtime["engine_kwargs"]["hf_overrides"] == {"architectures": ["Cosmos3ForConditionalGeneration"]}
    fresh_contract = normalized_caption_contract(resolve_model("cosmos3_nano"), "front_wide")
    assert fresh_contract["model_runtime"]["prepare_model_config_kwargs"]["hf_overrides"] == {
        "architectures": ["Cosmos3ForConditionalGeneration"]
    }


def test_terminal_row_repeats_episode_and_selected_camera_bindings() -> None:
    """Terminal metadata repeats and validates its enclosing episode evidence."""
    model = resolve_model("qwen3_8_27b_fp8")
    digest = caption_contract_digest(model, "front_wide")
    source = {
        "fragment_id": 4,
        "row_offset": 2,
        "episode_id": "episode",
        "materialization_digest": "materialization",
        "camera_key": "front_wide",
        "media_location": "/front-wide.mp4",
        "media_size_bytes": 123,
        "media_sha256": "a" * 64,
    }
    row = build_terminal_result_row(
        source,
        model=model,
        digest=digest,
        caption="caption",
        status="success",
        prompt_token_count=10,
        generated_token_count=20,
        error_type=None,
        error_message=None,
    )

    assert pa.Table.from_pylist([row], schema=staged_result_schema(model.key)).num_rows == 1
    validate_terminal_row(row, model=model, camera_key="front_wide", digest=digest)
    _, metadata_name = result_field_names(model.key)
    assert row[metadata_name]["episode_id"] == source["episode_id"]
    assert row[metadata_name]["materialization_digest"] == source["materialization_digest"]
    assert row[metadata_name]["media_sha256"] == source["media_sha256"]

    row[metadata_name]["episode_id"] = "different"
    with pytest.raises(ValueError, match="does not match"):
        validate_terminal_row(row, model=model, camera_key="front_wide", digest=digest)

    row[metadata_name]["episode_id"] = source["episode_id"]
    row[metadata_name]["caption_schema_version"] = True
    with pytest.raises(TypeError, match="caption_schema_version"):
        validate_terminal_row(row, model=model, camera_key="front_wide", digest=digest)
