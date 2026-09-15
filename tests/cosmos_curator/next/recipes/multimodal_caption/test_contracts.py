# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the versioned episode ``video_only`` result contract."""

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    build_terminal_result_row,
    caption_contract_digest,
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
from cosmos_curator.next.recipes.video_caption.contracts import resolve_model_spec
from cosmos_curator.next.recipes.video_caption.inference import caption_llm_preprocess


@pytest.mark.parametrize(
    ("model_key", "caption_name"),
    [
        ("qwen3_8_27b_fp8", "caption_video_only_qwen3_8_27b_fp8_v1"),
        ("qwen3_8_27b", "caption_video_only_qwen3_8_27b_v1"),
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


@pytest.mark.parametrize(
    ("model_key", "expected_digest"),
    [
        ("qwen3_8_27b_fp8", "0fa3f137b0b415ea49b338bd1eafa6af293566ead73340f985fdd82f0dbef2d4"),
        ("qwen3_8_27b", "e4c306c234aa8b7cc255b0c99da02a559936ca9ca5eed49995ee7267c94b1c43"),
    ],
)
def test_contract_digest_is_pinned_and_binds_camera(model_key: str, expected_digest: str) -> None:
    """Each model/camera contract has a pinned digest and changes with the camera."""
    model = resolve_model(model_key)
    digest = caption_contract_digest(model, "front_wide")
    other_camera_digest = caption_contract_digest(model, "front_left")

    assert digest == expected_digest
    assert digest != other_camera_digest
    contract = normalized_caption_contract(model, "front_wide")
    assert contract["episode_contract"]["camera_key"] == "front_wide"
    assert contract["video"] == {
        "container": "exact_episode_camera_mp4",
        "num_frames": -1,
        "fps": 2,
        "do_sample_frames": False,
        "longest_edge_override": None,
        "audio": "ignored",
    }


def test_shared_sampling_matches_runtime_and_both_contracts() -> None:
    """The shared request and both result digests use one sampling contract."""
    request = caption_llm_preprocess({"video_data_url": "data:video/mp4;base64,AAAA", "download_error_type": None})
    clip_contract = normalized_clip_caption_contract(resolve_model_spec("qwen3_8_27b_fp8"))
    episode_contract = normalized_caption_contract(resolve_model("qwen3_8_27b_fp8"), "front_wide")

    assert clip_contract["sampling"] == episode_contract["sampling"] == request["sampling_params"]


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
