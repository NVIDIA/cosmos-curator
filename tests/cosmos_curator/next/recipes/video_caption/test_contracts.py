# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pinned model, Arrow, digest, and terminal-state contracts."""

from pathlib import Path

import pyarrow as pa
import pytest

from cosmos_curator.core.utils import environment
from cosmos_curator.next.enrichment.contracts import FIELD_SET_METADATA, OWNER_METADATA
from cosmos_curator.next.recipes.video_caption.contracts import (
    DEFAULT_PROMPT,
    MAX_OUTPUT_TOKENS,
    CaptionModelSpec,
    caption_contract_digest,
    normalized_caption_contract,
    resolve_model_spec,
    terminal_metadata,
    validate_model_directory,
    validate_terminal_batch,
    validate_terminal_value,
)


@pytest.mark.parametrize(
    ("variant", "model_id", "revision", "precision", "digest"),
    [
        (
            "qwen3_8_27b_fp8",
            "Qwen/Qwen3.8-27B-FP8",
            "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            "FP8",
            "81b37950d8cac0c5cc02735c6f854df99c1a03fbb9df42a83963e9d6981f2cba",
        ),
        (
            "qwen3_8_27b",
            "Qwen/Qwen3.8-27B",
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "BF16",
            "c61d550a3870c76261fcd0b143654c579f1132f16655a39199d2af2087cb5f86",
        ),
    ],
)
def test_model_resolution_and_contract_digest_are_pinned(
    variant: str,
    model_id: str,
    revision: str,
    precision: str,
    digest: str,
) -> None:
    """Both allowed variants resolve to immutable identities and digests."""
    spec = resolve_model_spec(variant)  # type: ignore[arg-type]

    assert (spec.model_id, spec.revision, spec.precision) == (model_id, revision, precision)
    assert caption_contract_digest(spec) == digest


def test_max_model_len_is_result_defining_only_when_set(caption_spec: CaptionModelSpec) -> None:
    """A context limit can end a caption early, so a set limit changes the digest; null leaves it as is."""
    default_digest = caption_contract_digest(caption_spec)

    assert "model_runtime" not in normalized_caption_contract(caption_spec)
    assert caption_contract_digest(caption_spec, max_model_len=None) == default_digest
    assert normalized_caption_contract(caption_spec, max_model_len=32_768)["model_runtime"] == {"max_model_len": 32_768}
    assert caption_contract_digest(caption_spec, max_model_len=32_768) != default_digest
    assert caption_contract_digest(caption_spec, max_model_len=32_768) != caption_contract_digest(
        caption_spec, max_model_len=65_536
    )


def test_arrow_field_set_is_exact_and_owned(caption_spec: CaptionModelSpec) -> None:
    """Both nullable top-level fields carry exact ownership and child schemas."""
    caption, metadata = caption_spec.fields

    assert caption.type == pa.large_string()
    assert caption.nullable is True
    assert pa.types.is_struct(metadata.type)
    assert metadata.nullable is True
    assert (
        caption.metadata
        == metadata.metadata
        == {
            OWNER_METADATA: b"video-caption",
            FIELD_SET_METADATA: caption_spec.caption_field_name.encode(),
        }
    )
    assert [(child.name, child.nullable) for child in metadata.type] == [
        ("caption_schema_version", False),
        ("contract_digest", False),
        ("status", False),
        ("model_id", False),
        ("model_revision", False),
        ("prompt", False),
        ("prompt_token_count", True),
        ("generated_token_count", True),
        ("error_type", True),
        ("error_message", True),
    ]


def test_normalized_contract_contains_every_result_defining_media_setting(caption_spec: CaptionModelSpec) -> None:
    """The digest input captures prompt, sampling, and video processing semantics."""
    contract = normalized_caption_contract(caption_spec)

    assert MAX_OUTPUT_TOKENS == 2_048
    assert contract["message"][0]["content"][0]["type"] == "video_url"
    assert contract["message"][0]["content"][1] == {"type": "text", "text": DEFAULT_PROMPT}
    assert contract["chat_template"]["kwargs"] == {"enable_thinking": False}
    assert contract["sampling"] == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    assert contract["video"] == {
        "container": "exact_clip_mp4",
        "num_frames": -1,
        "fps": 2,
        "do_sample_frames": False,
        "longest_edge_override": None,
        "audio": "ignored",
    }
    assert contract["token_counts"] == {
        "prompt_token_count": {"type": "integer", "nullable": True, "minimum": 0},
        "generated_token_count": {"type": "integer", "nullable": True, "minimum": 0},
    }
    assert contract["failure_policy"] == {
        "item_media": "terminal_error",
        "multimodal_preparation": "fail_phase",
        "inference": "fail_phase",
        "shared_runtime_or_storage": "fail_phase",
    }
    assert contract["terminal_states"] == {
        "success": {
            "caption": "non_null_string",
            "error_type": "null",
            "error_message": "null",
            "generated_token_count_if_present": {"less_than": MAX_OUTPUT_TOKENS},
        },
        "truncated": {
            "caption": "non_null_string",
            "error_type": "null",
            "error_message": "null",
            "generated_token_count_if_present": {"greater_than_or_equal": MAX_OUTPUT_TOKENS},
        },
        "error": {
            "caption": "null",
            "error_type": "non_empty_string",
            "error_message": "non_empty_string",
        },
    }


@pytest.mark.parametrize("status", ["success", "truncated"])
def test_success_terminal_states_validate(
    status: str,
    caption_spec: CaptionModelSpec,
    caption_digest: str,
) -> None:
    """Successful and token-limited terminal values satisfy the contract."""
    generated = MAX_OUTPUT_TOKENS if status == "truncated" else 12
    metadata = terminal_metadata(
        caption_spec,
        caption_digest,
        status=status,
        prompt_token_count=8,
        generated_token_count=generated,
        error_type=None,
        error_message=None,
    )

    validate_terminal_value("a caption", metadata, spec=caption_spec, digest=caption_digest)


def test_error_terminal_state_validates(caption_spec: CaptionModelSpec, caption_digest: str) -> None:
    """A per-item error is terminal only with null caption and error detail."""
    metadata = terminal_metadata(
        caption_spec,
        caption_digest,
        status="error",
        prompt_token_count=None,
        generated_token_count=None,
        error_type="DecodeError",
        error_message="bad MP4",
    )

    validate_terminal_value(None, metadata, spec=caption_spec, digest=caption_digest)


@pytest.mark.parametrize(
    ("caption", "changes", "match"),
    [
        (None, {"status": "success"}, "non-null caption"),
        ("caption", {"status": "error", "error_type": "X", "error_message": "bad"}, "null caption"),
        (None, {"status": "error", "error_type": None, "error_message": "bad"}, "error_type"),
        (None, {"status": "error", "error_type": "X", "error_message": ""}, "error_type/error_message"),
        ("caption", {"contract_digest": "wrong"}, "contract_digest"),
        ("caption", {"caption_schema_version": True}, "caption_schema_version"),
        ("caption", {"generated_token_count": -1}, "non-negative"),
        ("caption", {"status": "success", "generated_token_count": MAX_OUTPUT_TOKENS}, "truncated"),
    ],
)
def test_invalid_terminal_states_are_rejected(
    caption: str | None,
    changes: dict[str, object],
    match: str,
    caption_spec: CaptionModelSpec,
    caption_digest: str,
) -> None:
    """Mismatched digests and inconsistent status payloads are rejected."""
    metadata = (
        terminal_metadata(
            caption_spec,
            caption_digest,
            status="success",
            prompt_token_count=1,
            generated_token_count=2,
            error_type=None,
            error_message=None,
        )
        | changes
    )

    with pytest.raises((TypeError, ValueError), match=match):
        validate_terminal_value(caption, metadata, spec=caption_spec, digest=caption_digest)


def test_batch_validation_rejects_any_non_terminal_row(caption_spec: CaptionModelSpec, caption_digest: str) -> None:
    """Shared publication validates Arrow batches; one pending or stale row rejects the batch."""
    metadata = terminal_metadata(
        caption_spec,
        caption_digest,
        status="success",
        prompt_token_count=1,
        generated_token_count=2,
        error_type=None,
        error_message=None,
    )
    schema = pa.schema([pa.field("clip_id", pa.string()), *caption_spec.fields])
    rows = [
        {"clip_id": "a", caption_spec.caption_field_name: "caption", caption_spec.metadata_field_name: metadata},
        {"clip_id": "b", caption_spec.caption_field_name: "caption", caption_spec.metadata_field_name: metadata},
    ]
    validate_terminal_batch(pa.RecordBatch.from_pylist(rows, schema=schema), spec=caption_spec, digest=caption_digest)

    rows[1][caption_spec.metadata_field_name] = None
    with pytest.raises(ValueError, match="must not be null"):
        validate_terminal_batch(
            pa.RecordBatch.from_pylist(rows, schema=schema), spec=caption_spec, digest=caption_digest
        )
    rows[1][caption_spec.metadata_field_name] = metadata | {"contract_digest": "0" * 64}
    with pytest.raises(ValueError, match="contract_digest"):
        validate_terminal_batch(
            pa.RecordBatch.from_pylist(rows, schema=schema), spec=caption_spec, digest=caption_digest
        )


def test_model_directory_must_contain_config_and_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caption_spec: CaptionModelSpec,
) -> None:
    """Runtime setup accepts complete pre-staged model files without downloader metadata."""
    monkeypatch.setattr(environment, "CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR", tmp_path)
    model_dir = caption_spec.runtime_model_dir
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        validate_model_directory(caption_spec)

    (model_dir / "model.safetensors").write_bytes(b"weights")
    assert validate_model_directory(caption_spec) == model_dir
