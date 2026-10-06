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

"""Tests for video-vector field identity and terminal row invariants."""

from copy import deepcopy
from pathlib import Path
from typing import get_args

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.recipes.multimodal_embedding.config import ModelVariant, resolve_config_data
from cosmos_curator.next.recipes.multimodal_embedding.contracts import (
    EmbeddingTerminalResult,
    build_terminal_result_row,
    embedding_contract_digest,
    embedding_contract_digest_from_config,
    logical_result_fields,
    normalized_embedding_contract,
    resolve_model,
    result_field_names,
    result_fields,
    staged_result_schema,
    supported_model_variants,
    validate_field_set,
    validate_terminal_batch,
    validate_terminal_row,
)
from cosmos_curator.next.recipes.result_keys import camera_result_key


def test_camera_namespaces_are_bounded_and_distinct() -> None:
    """Effective keys preserve the camera spelling used by multimodal caption tagging."""
    assert camera_result_key("default", "front_wide") == "default_front_wide"
    assert camera_result_key("default", "left_fisheye") == "default_left_fisheye"
    assert camera_result_key("a" * 42, "front") == "a" * 42 + "_front"


@pytest.mark.parametrize("camera", ["", "Front-Wide", "front wide", "front-wide", "rear/left", "x" * 41])
def test_camera_namespaces_reject_unsafe_or_overlong_effective_keys(camera: str) -> None:
    """Camera identities are never silently renamed, hashed, or truncated."""
    with pytest.raises(ValueError, match="Invalid effective result key"):
        camera_result_key("default", camera)


def test_camera_contract_disambiguates_concatenated_namespaces() -> None:
    """The exact spelling can coincide across base keys; the camera contract still differs."""
    model = resolve_model("cosmos_embed1_336p")
    first = camera_result_key("a_b", "c")
    second = camera_result_key("a", "b_c")
    assert first == second
    assert embedding_contract_digest(model, camera_key="c", result_key=first) != embedding_contract_digest(
        model, camera_key="b_c", result_key=second
    )


def _source() -> dict[str, object]:
    return {
        "episode_id": "episode-1",
        "materialization_digest": "materialization",
        "camera_key": "front_wide",
        "media_sha256": "a" * 64,
        "start_ns": 10,
        "exclusive_end_ns": 1_000_000_010,
    }


def test_result_fields_are_fixed_width_float32_with_a_metadata_sibling() -> None:
    """One result key owns a complete, digest-tagged field set."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    embedding_name, metadata_name = result_field_names("default", model_key=model.key)
    fields = result_fields("default", model_key=model.key, digest=digest)

    assert [field.name for field in fields] == [embedding_name, metadata_name]
    assert pa.types.is_fixed_size_list(fields.field(embedding_name).type)
    assert fields.field(embedding_name).type.list_size == model.vector_dimension
    assert fields.field(embedding_name).type.value_type == pa.float32()
    assert fields.field(embedding_name).metadata[b"cosmos_curator.contract_digest"] == digest.encode()
    assert fields.field(metadata_name).metadata[b"cosmos_curator.contract_digest"] == digest.encode()


def test_null_staged_vector_roundtrips_through_parquet(tmp_path: Path) -> None:
    """An item error remains readable before conversion to Lance's fixed-width field."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    source = {
        **_source(),
        "fragment_id": 0,
        "row_offset": 0,
        "media_location": str(tmp_path / "missing.mp4"),
        "media_size_bytes": 1,
    }
    result = EmbeddingTerminalResult(
        status="error",
        vector=None,
        max_episode_duration_s=60.0,
        error_code="media_object_error",
        error_message="media missing",
    )
    vector_name, _ = result_field_names("default", model_key=model.key)
    staged = pa.Table.from_pylist(
        [
            {
                **source,
                **build_terminal_result_row(source, model=model, result_key="default", digest=digest, result=result),
            }
        ],
        schema=staged_result_schema("default", model_key=model.key, digest=digest),
    )
    path = tmp_path / "error.parquet"
    pq.write_table(staged, path)
    assert pq.read_table(path)[vector_name].to_pylist() == [None]


@pytest.mark.parametrize(
    ("variant", "resolution", "dimension"),
    [
        ("cosmos_embed1_224p", 224, 256),
        ("cosmos_embed1_336p", 336, 768),
        ("cosmos_embed1_448p", 448, 768),
    ],
)
def test_model_binding_drives_the_public_field_identity(variant: str, resolution: int, dimension: int) -> None:
    """Every selectable Cosmos checkpoint defines its processor and vector shape."""
    assert supported_model_variants() == tuple(sorted(get_args(ModelVariant)))
    model = resolve_model(variant)
    assert model.video_frame_count == 8
    assert model.resolution == resolution
    assert model.vector_dimension == dimension
    assert model.processor_revision == model.revision
    assert result_field_names("default", model_key=model.key)[0] == f"embedding_video_default_ce1_{resolution}p_v1"
    assert logical_result_fields("default", model_key=model.key).field(0).type.list_size == dimension
    with pytest.raises(ValueError, match="Unsupported multimodal embedding model"):
        resolve_model("internvideo2_mm")


def test_vector_contract_excludes_duration_and_retry_policy(tmp_path: Path) -> None:
    """Eligibility changes do not force a new result field or re-embed eligible rows."""
    config_payload = {
        "schema_version": 1,
        "kind": "multimodal-embedding",
        "input": {"episodes_lance_uri": str(tmp_path / "episodes.lance")},
        "video": {"camera_keys": ["front_wide"]},
    }
    baseline = resolve_config_data(config_payload)
    adjusted = deepcopy(config_payload)
    adjusted["selection"] = {"max_episode_duration_s": 120.0, "retry_errors": True}
    adjusted["execution"] = {
        "media_concurrency": 2,
        "decode_concurrency": 3,
        "inference_batch_size": 8,
        "inference_gpus": 0.5,
        "media_attempts": 5,
        "decode_cpus": 4,
        "decode_memory_gib": 8.0,
        "lance_read_batch_size": 32,
    }
    assert embedding_contract_digest_from_config(
        baseline, camera_key="front_wide"
    ) == embedding_contract_digest_from_config(resolve_config_data(adjusted), camera_key="front_wide")

    changed_camera = deepcopy(config_payload)
    changed_camera["video"] = {"camera_keys": ["front_left"]}
    assert embedding_contract_digest_from_config(
        baseline, camera_key="front_wide"
    ) != embedding_contract_digest_from_config(resolve_config_data(changed_camera), camera_key="front_left")
    contract = normalized_embedding_contract(
        resolve_model(baseline.model.variant), camera_key="front_wide", result_key="default"
    )
    assert "decode" not in contract["sampling"]


def test_reusing_a_result_key_requires_the_complete_exact_field_set() -> None:
    """A partial or different vector field set cannot be mistaken for a compatible result."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    fields = result_fields("default", model_key=model.key, digest=digest)
    validate_field_set(list(fields), result_key="default", model_key=model.key, digest=digest)

    with pytest.raises(ValueError, match="incomplete"):
        validate_field_set([fields.field(0)], result_key="default", model_key=model.key, digest=digest)

    changed = fields.field(0).with_type(pa.list_(pa.float32(), model.vector_dimension - 1))
    with pytest.raises(ValueError, match="incompatible"):
        validate_field_set([changed, fields.field(1)], result_key="default", model_key=model.key, digest=digest)

    for index in range(len(fields)):
        changed_fields = list(fields)
        changed_fields[index] = changed_fields[index].with_metadata(
            {**(changed_fields[index].metadata or {}), b"cosmos_curator.contract_digest": b"incompatible"}
        )
        with pytest.raises(ValueError, match="incompatible"):
            validate_field_set(changed_fields, result_key="default", model_key=model.key, digest=digest)


def test_terminal_rows_bind_the_episode_camera_and_media_identity() -> None:
    """Success and terminal non-success rows have unambiguous nullable-vector semantics."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    success = build_terminal_result_row(
        _source(),
        model=model,
        result_key="default",
        digest=digest,
        result=EmbeddingTerminalResult(
            status="success",
            vector=[1.0 / model.vector_dimension**0.5] * model.vector_dimension,
            max_episode_duration_s=60.0,
            error_code=None,
            error_message=None,
        ),
    )
    _embedding_name, metadata_name = result_field_names("default", model_key=model.key)
    assert success[metadata_name]["episode_duration_ns"] == 1_000_000_000  # type: ignore[index]

    long_source = {**_source(), "exclusive_end_ns": 120_000_000_010}
    unsupported = build_terminal_result_row(
        long_source,
        model=model,
        result_key="default",
        digest=digest,
        result=EmbeddingTerminalResult(
            status="unsupported_duration",
            vector=None,
            max_episode_duration_s=60.0,
            error_code="unsupported_duration",
            error_message=None,
        ),
    )
    validate_terminal_row(unsupported, model=model, camera_key="front_wide", result_key="default", digest=digest)

    invalid = dict(unsupported)
    invalid[result_field_names("default", model_key=model.key)[0]] = [0.0] * model.vector_dimension
    with pytest.raises(ValueError, match="null vector"):
        validate_terminal_row(invalid, model=model, camera_key="front_wide", result_key="default", digest=digest)


def test_terminal_validation_checks_contract_norm_and_duration_eligibility() -> None:
    """Published rows cannot be accepted under changed semantics or corrupt outcome data."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    vector_name, metadata_name = result_field_names("default", model_key=model.key)
    row = build_terminal_result_row(
        _source(),
        model=model,
        result_key="default",
        digest=digest,
        result=EmbeddingTerminalResult(
            status="success",
            vector=np.full(model.vector_dimension, 1 / np.sqrt(model.vector_dimension), dtype=np.float32),
            max_episode_duration_s=60.0,
            error_code=None,
            error_message=None,
            media_duration_ns=1_000_000_000,
        ),
    )
    for field, replacement in (
        ("contract_digest", "old-digest"),
        ("result_key", "other"),
        ("model_revision", "other-revision"),
        ("sampling_version", "other-sampling"),
    ):
        changed = deepcopy(row)
        changed[metadata_name][field] = replacement
        with pytest.raises(ValueError, match=field):
            validate_terminal_row(changed, model=model, camera_key="front_wide", result_key="default", digest=digest)
    non_unit = deepcopy(row)
    non_unit[vector_name] = [0.5] * model.vector_dimension
    with pytest.raises(ValueError, match="unit norm"):
        validate_terminal_row(non_unit, model=model, camera_key="front_wide", result_key="default", digest=digest)
    array_row = deepcopy(row)
    array_row[vector_name] = np.asarray(row[vector_name], dtype=np.float32)
    validate_terminal_row(array_row, model=model, camera_key="front_wide", result_key="default", digest=digest)

    long_source = {**_source(), "exclusive_end_ns": 120_000_000_010}
    unsupported = build_terminal_result_row(
        long_source,
        model=model,
        result_key="default",
        digest=digest,
        result=EmbeddingTerminalResult(
            status="unsupported_duration",
            vector=None,
            max_episode_duration_s=60.0,
            error_code="unsupported_duration",
            error_message=None,
        ),
    )
    for maximum in (None, 120.0):
        changed = deepcopy(unsupported)
        changed[metadata_name]["max_episode_duration_s"] = maximum
        with pytest.raises(ValueError, match="recorded maximum"):
            validate_terminal_row(changed, model=model, camera_key="front_wide", result_key="default", digest=digest)


def test_terminal_batch_rejects_swapped_vector_nullability() -> None:
    """Equal aggregate null counts cannot hide a null success and a populated error."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    vector_name, _ = result_field_names("default", model_key=model.key)
    success = build_terminal_result_row(
        _source(),
        model=model,
        result_key="default",
        digest=digest,
        result=EmbeddingTerminalResult(
            status="success",
            vector=np.full(model.vector_dimension, 1 / np.sqrt(model.vector_dimension), dtype=np.float32),
            max_episode_duration_s=60.0,
            error_code=None,
            error_message=None,
            media_duration_ns=1_000_000_000,
        ),
    )
    failure = build_terminal_result_row(
        {**_source(), "episode_id": "episode-2"},
        model=model,
        result_key="default",
        digest=digest,
        result=EmbeddingTerminalResult(
            status="error",
            vector=None,
            max_episode_duration_s=60.0,
            error_code="media_object_error",
            error_message="media missing",
        ),
    )
    fields = result_fields("default", model_key=model.key, digest=digest)

    def check(rows: list[dict[str, object]]) -> None:
        validate_terminal_batch(
            pa.Table.from_pylist(rows, schema=fields),
            model=model,
            camera_key="front_wide",
            result_key="default",
            digest=digest,
        )

    check([success, failure])
    swapped = [deepcopy(success), deepcopy(failure)]
    swapped[0][vector_name], swapped[1][vector_name] = swapped[1][vector_name], swapped[0][vector_name]
    with pytest.raises(ValueError, match="row 0 has vector nullability"):
        check(swapped)


def test_duration_mismatch_error_records_observed_media_duration() -> None:
    """The v1 schema reserves media duration and a distinct mismatch error."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    row = build_terminal_result_row(
        _source(),
        model=model,
        result_key="default",
        digest=digest,
        result=EmbeddingTerminalResult(
            status="error",
            vector=None,
            max_episode_duration_s=60.0,
            error_code="duration_mismatch",
            error_message="media span differs",
            media_duration_ns=2_000_000_000,
        ),
    )
    metadata_name = result_field_names("default", model_key=model.key)[1]
    assert (
        result_fields("default", model_key=model.key, digest=digest)
        .field(metadata_name)
        .type.field("media_duration_ns")
        .nullable
    )
    assert row[metadata_name]["media_duration_ns"] == 2_000_000_000
    invalid = deepcopy(row)
    invalid[metadata_name]["media_duration_ns"] = None
    with pytest.raises(ValueError, match="requires media_duration_ns"):
        validate_terminal_row(invalid, model=model, camera_key="front_wide", result_key="default", digest=digest)


def test_logical_fields_do_not_contain_the_digest_used_to_register_them() -> None:
    """Digest derivation cannot become circular through the Arrow field metadata."""
    assert all(
        b"cosmos_curator.contract_digest" not in (field.metadata or {})
        for field in logical_result_fields("default", model_key="cosmos_embed1_336p")
    )


@pytest.mark.parametrize(("norm", "valid"), [(0.99609375, True), (1.019, True), (1.021, False)])
def test_row_and_batch_share_bf16_norm_tolerance(norm: float, *, valid: bool) -> None:
    """Rounding-tolerant validation agrees in initial publication and no-op scans."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    vector_name, _ = result_field_names("default", model_key=model.key)
    vector = np.zeros(model.vector_dimension, dtype=np.float32)
    vector[0] = 1
    row = build_terminal_result_row(
        _source(),
        model=model,
        result_key="default",
        digest=digest,
        result=EmbeddingTerminalResult(
            status="success",
            vector=vector,
            max_episode_duration_s=60,
            error_code=None,
            error_message=None,
            media_duration_ns=1_000_000_000,
        ),
    )
    row[vector_name][0] = norm
    table = pa.Table.from_pylist([row], schema=result_fields("default", model_key=model.key, digest=digest))
    if valid:
        validate_terminal_row(row, model=model, camera_key="front_wide", result_key="default", digest=digest)
        validate_terminal_batch(table, model=model, camera_key="front_wide", result_key="default", digest=digest)
    else:
        with pytest.raises(ValueError, match="unit norm"):
            validate_terminal_row(row, model=model, camera_key="front_wide", result_key="default", digest=digest)
        with pytest.raises(ValueError, match="unit norm"):
            validate_terminal_batch(table, model=model, camera_key="front_wide", result_key="default", digest=digest)


@pytest.mark.parametrize("result_key", ["", "Uppercase", "1leading", "with-dash", "a" * 49])
def test_result_field_names_reject_invalid_keys(result_key: str) -> None:
    """Contract callers cannot bypass result-key validation through the naming helper."""
    with pytest.raises(ValueError, match="Invalid embedding result key"):
        result_field_names(result_key, model_key="cosmos_embed1_336p")


def test_result_field_names_accept_maximum_length_key() -> None:
    """The contract naming boundary matches the shared 48-character result-key limit."""
    key = "a" * 48
    vector, metadata = result_field_names(key, model_key="cosmos_embed1_336p")
    assert vector == f"embedding_video_{key}_ce1_336p_v1"
    assert metadata == f"{vector}_metadata"
