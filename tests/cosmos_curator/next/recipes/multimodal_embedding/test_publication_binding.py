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

"""Check that the caption publication seam can carry an Arrow video vector."""

from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.multimodal_embedding.contracts import (
    CameraEmbeddingPlan,
    EmbeddingRunPlan,
    EmbeddingTerminalResult,
    build_terminal_result_row,
    embedding_contract_digest,
    resolve_model,
    result_field_names,
    result_fields,
    validate_terminal_row,
)
from cosmos_curator.next.recipes.multimodal_embedding.publication import publication_bindings
from cosmos_curator.next.recipes.video_caption.bindings import (
    CaptionInputBindings,
    CaptionPublicationBindings,
    CaptionResultBindings,
    CaptionRunBindings,
)
from cosmos_curator.next.recipes.video_caption.lance_state import FragmentCaptionState, classify_fragment
from cosmos_curator.next.recipes.video_caption.publication import commit_prepared_update, prepare_fragment_update
from cosmos_curator.next.utils.identity import canonical_digest


def test_explicit_run_plan_preserves_identity_and_validates_every_camera(tmp_path: Path) -> None:
    """Aggregate guards have episode identity and cannot silently validate only their workspace anchor."""
    model = resolve_model("cosmos_embed1_336p")
    cameras = tuple(
        CameraEmbeddingPlan.resolve(model, camera_key=key, result_key=f"default_{key}")
        for key in ("rear", "front_wide")
    )
    uri = str(tmp_path / "episodes.lance")
    plan = EmbeddingRunPlan(uri, "default", model, cameras, {0: ("rear",)})
    assert plan.digest == canonical_digest(
        {"multi_camera_version": 1, "cameras": {camera.camera_key: camera.digest for camera in cameras}}
    )
    reordered = EmbeddingRunPlan(uri, "default", model, tuple(reversed(cameras)), {})
    assert reordered.digest == plan.digest
    assert reordered.workspace_field == plan.workspace_field
    assert plan.fields.names == [name for camera in plan.cameras for name in camera.fields.names]
    bindings = publication_bindings(plan)
    assert not {"camera_key", "media_location", "media_sha256"} & set(bindings.input.mapped_schema.names)
    with pytest.raises(TypeError, match="requires per-camera terminal rows"):
        bindings.result.validate_terminal_value(None, None)
    row = {}
    for camera in cameras:
        source = {
            "episode_id": "episode-1",
            "materialization_digest": "materialization",
            "camera_key": camera.camera_key,
            "media_sha256": "a" * 64,
            "start_ns": 0,
            "exclusive_end_ns": 1_000_000_000,
        }
        row.update(
            build_terminal_result_row(
                source,
                model=model,
                result_key=camera.result_key,
                digest=camera.digest,
                result=EmbeddingTerminalResult(
                    status="success",
                    vector=np.eye(1, model.vector_dimension, dtype=np.float32)[0],
                    max_episode_duration_s=60.0,
                    error_code=None,
                    error_message=None,
                ),
            )
        )
    batch = pa.Table.from_pylist([row], schema=plan.fields)
    assert bindings.result.validate_terminal_batch is not None
    bindings.result.validate_terminal_batch(batch)
    rear = next(camera for camera in cameras if camera.camera_key == "rear")
    row[rear.metadata_field]["contract_digest"] = "wrong-rear-contract"
    with pytest.raises(ValueError, match="contract_digest"):
        bindings.result.validate_terminal_batch(pa.Table.from_pylist([row], schema=plan.fields))


def test_run_plan_rejects_colliding_camera_fields(tmp_path: Path) -> None:
    """An aggregate plan cannot accidentally give two cameras the same namespace."""
    model = resolve_model("cosmos_embed1_336p")
    cameras = tuple(
        CameraEmbeddingPlan.resolve(model, camera_key=key, result_key="default") for key in ("front", "rear")
    )
    with pytest.raises(ValueError, match="overlapping embedding fields"):
        EmbeddingRunPlan(str(tmp_path / "episodes.lance"), "default", model, cameras, {})


def test_fixed_size_vector_passes_through_fragment_update_and_classification(tmp_path: Path) -> None:
    """A fixed-size list works at the binding, Arrow update, and completion boundaries."""
    model = resolve_model("cosmos_embed1_336p")
    digest = embedding_contract_digest(model, camera_key="front_wide", result_key="default")
    vector_name, metadata_name = result_field_names("default", model_key=model.key)
    fields = result_fields("default", model_key=model.key, digest=digest)
    row_id_field = pa.field("episode_id", pa.string(), nullable=False)
    staged_schema = pa.schema([row_id_field, *fields])
    source = {
        "episode_id": "episode-1",
        "materialization_digest": "materialization",
        "camera_key": "front_wide",
        "media_sha256": "a" * 64,
        "start_ns": 0,
        "exclusive_end_ns": 1_000_000_000,
    }
    terminal = build_terminal_result_row(
        source,
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

    def validate_value(vector: object, metadata: dict[str, Any] | None) -> None:
        validate_terminal_row(
            {vector_name: vector, metadata_name: metadata},
            model=model,
            camera_key="front_wide",
            result_key="default",
            digest=digest,
        )

    bindings = CaptionRunBindings(
        input=CaptionInputBindings(
            source_columns=("episode_id",),
            source_schema=pa.schema([row_id_field]),
            mapped_schema=pa.schema([row_id_field]),
            project_batch=lambda batch: batch,
            row_id_column="episode_id",
            media_uri_column="media_location",
            media_size_column="media_size_bytes",
            media_sha256_column="media_sha256",
            staged_binding_columns=("episode_id",),
        ),
        result=CaptionResultBindings(
            caption_field_name=vector_name,
            metadata_field_name=metadata_name,
            fields=fields,
            staged_schema=staged_schema,
            digest=digest,
            build_terminal_row=lambda *_args, **_kwargs: terminal,
            validate_terminal_value=validate_value,
            validate_staged_row=lambda *_args, **_kwargs: None,
        ),
        publication=CaptionPublicationBindings(
            validate_base_schema=lambda *_args: None,
            protected_field_names=("episode_id",),
            base_field_owner="multimodal-split",
            transaction_properties={},
        ),
        prompt_template="",
    )
    uri = str(tmp_path / "episodes.lance")
    pending = pa.Table.from_pylist(
        [{"episode_id": source["episode_id"], vector_name: None, metadata_name: None}], schema=staged_schema
    )
    dataset = lance.write_dataset(pending, uri, data_storage_version="2.2")
    fragment = dataset.get_fragments()[0]
    assert classify_fragment(fragment, bindings=bindings) is FragmentCaptionState.PENDING

    staged = pa.Table.from_pylist([{"episode_id": source["episode_id"], **terminal}], schema=staged_schema)
    descriptor = prepare_fragment_update(dataset, fragment, staged, bindings)
    commit_prepared_update(uri, descriptor, bindings, storage_options=None)

    published = lance.dataset(uri)
    assert classify_fragment(published.get_fragments()[0], bindings=bindings) is FragmentCaptionState.COMPLETE
    result = published.to_table(columns=[vector_name, metadata_name]).to_pylist()[0]
    assert len(result[vector_name]) == model.vector_dimension
    assert result[metadata_name]["status"] == "success"
