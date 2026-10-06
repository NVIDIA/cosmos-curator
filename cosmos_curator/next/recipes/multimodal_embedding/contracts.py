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

"""Video-vector identity, Arrow fields, and terminal-result invariants."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from cosmos_curator.core.utils import environment
from cosmos_curator.models.registry import resolve_registered_model, validate_model_directory
from cosmos_curator.next.recipes.multimodal_embedding.config import ResolvedMultimodalEmbeddingConfig
from cosmos_curator.next.recipes.multimodal_embedding.episode_input import (
    EPISODE_MAPPED_INPUT_SCHEMA,
    EPISODE_PUBLICATION_SCHEMA,
)
from cosmos_curator.next.recipes.result_keys import is_valid_result_key
from cosmos_curator.next.utils.identity import canonical_digest

EMBEDDING_SCHEMA_VERSION: Final = 1
RESULT_VERSION: Final = 1
NORMALIZATION_VERSION: Final = "model_bf16_l2_cast_float32_v1"
OWNER_METADATA: Final = b"cosmos_curator.owner"
FIELD_SET_METADATA: Final = b"cosmos_curator.field_set"
CONTRACT_DIGEST_METADATA: Final = b"cosmos_curator.contract_digest"
TERMINAL_STATUSES: Final = frozenset({"success", "error", "unsupported_duration"})
ITEM_ERROR_CODES: Final = frozenset(
    {
        "media_object_error",
        "size_mismatch",
        "sha256_mismatch",
        "invalid_timestamps",
        "decode_error",
        "invalid_vector",
        "duration_mismatch",
    }
)
VECTOR_NORM_TOLERANCE: Final = 0.02


@dataclass(frozen=True)
class VideoEmbeddingModelBinding:
    """All recipe-level facts needed to safely select one video embedder."""

    variant: str
    registry_key: str
    field_key: str
    runtime_environment: str
    vector_dimension: int
    video_frame_count: int
    sampling_version: str
    resolution: int


_MODEL_BINDINGS: Final = {
    f"cosmos_embed1_{resolution}p": VideoEmbeddingModelBinding(
        variant=f"cosmos_embed1_{resolution}p",
        registry_key=f"cosmos_embed1_{resolution}p",
        field_key=f"ce1_{resolution}p",
        runtime_environment="legacy-transformers",
        vector_dimension=dimension,
        video_frame_count=8,
        sampling_version="frame_linspace_endpoints_v1",
        resolution=resolution,
    )
    for resolution, dimension in ((224, 256), (336, 768), (448, 768))
}


def supported_model_variants() -> tuple[str, ...]:
    """Return all config-selectable variants in deterministic order."""
    return tuple(sorted(_MODEL_BINDINGS))


def resolve_model_binding(variant: str) -> VideoEmbeddingModelBinding:
    """Resolve one supported binding without importing a model runtime."""
    binding = _MODEL_BINDINGS.get(variant)
    if binding is None:
        msg = f"Unsupported multimodal embedding model {variant!r}. Supported: {sorted(_MODEL_BINDINGS)}"
        raise ValueError(msg)
    return binding


@dataclass(frozen=True)
class VideoEmbeddingModelSpec:
    """One model and processor revision which produces this recipe's vectors."""

    key: str
    field_key: str
    model_id: str
    revision: str
    precision: str
    processor_revision: str
    runtime_environment: str
    vector_dimension: int
    video_frame_count: int
    sampling_version: str
    resolution: int

    @property
    def runtime_model_dir(self) -> Path:
        """Return the identically mounted model path expected by every worker."""
        return environment.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR / self.model_id


@dataclass(frozen=True)
class CameraEmbeddingPlan:
    """One camera's explicit field ownership and immutable vector contract."""

    camera_key: str
    result_key: str
    digest: str
    vector_field: str
    metadata_field: str
    fields: pa.Schema
    staged_schema: pa.Schema

    @classmethod
    def resolve(cls, model: VideoEmbeddingModelSpec, *, camera_key: str, result_key: str) -> "CameraEmbeddingPlan":
        """Resolve one camera without inspecting table history or caption bindings."""
        digest = embedding_contract_digest(model, camera_key=camera_key, result_key=result_key)
        vector, metadata = result_field_names(result_key, model_key=model.key)
        return cls(
            camera_key,
            result_key,
            digest,
            vector,
            metadata,
            result_fields(result_key, model_key=model.key, digest=digest),
            staged_result_schema(result_key, model_key=model.key, digest=digest),
        )


@dataclass(frozen=True)
class EmbeddingRunPlan:
    """Shared episode publication state, with independently contracted camera plans."""

    episodes_lance_uri: str
    base_result_key: str
    model: VideoEmbeddingModelSpec
    cameras: tuple[CameraEmbeddingPlan, ...]
    pending_cameras: dict[int, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Canonicalize camera order and reject colliding ownership before execution."""
        cameras = tuple(sorted(self.cameras, key=lambda camera: camera.camera_key))
        if not cameras or len({camera.camera_key for camera in cameras}) != len(cameras):
            msg = "An embedding run requires distinct camera plans"
            raise ValueError(msg)
        names = [name for camera in cameras for name in camera.fields.names]
        if len(set(names)) != len(names):
            msg = "Camera plans cannot own overlapping embedding fields"
            raise ValueError(msg)
        if any(set(keys) - {camera.camera_key for camera in cameras} for keys in self.pending_cameras.values()):
            msg = "Pending camera work is outside the selected run plan"
            raise ValueError(msg)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "pending_cameras", dict(self.pending_cameras))

    @cached_property
    def digest(self) -> str:
        """Keep the existing combined recovery identity, independent of pending work."""
        return canonical_digest({"multi_camera_version": 1, "cameras": {c.camera_key: c.digest for c in self.cameras}})

    @cached_property
    def fields(self) -> pa.Schema:
        """Own every selected camera field, without a primary camera's result state."""
        return pa.schema([field for camera in self.cameras for field in camera.fields])

    @cached_property
    def staged_schema(self) -> pa.Schema:
        """Assemble camera results beside camera-independent episode identities."""
        return pa.schema(
            [
                *EPISODE_PUBLICATION_SCHEMA,
                *[
                    field
                    for camera in self.cameras
                    for field in list(camera.staged_schema)[len(EPISODE_MAPPED_INPUT_SCHEMA) :]
                ],
            ]
        )

    @property
    def workspace_field(self) -> str:
        """Retain the existing deterministic workspace anchor; it carries no camera state."""
        return self.cameras[0].vector_field


@dataclass(frozen=True)
class EmbeddingTerminalResult:
    """One terminal inference outcome before it is bound to an episode row."""

    status: str
    vector: Sequence[float] | NDArray[np.float32] | None
    max_episode_duration_s: float | None
    error_code: str | None
    error_message: str | None
    media_duration_ns: int | None = None


def resolve_model(model_key: str) -> VideoEmbeddingModelSpec:
    """Resolve one recipe-supported model binding from the model registry."""
    binding = resolve_model_binding(model_key)
    registered = resolve_registered_model(binding.registry_key)
    return VideoEmbeddingModelSpec(
        key=model_key,
        field_key=binding.field_key,
        model_id=registered.model_id,
        revision=registered.revision,
        precision=registered.precision,
        processor_revision=registered.revision,
        runtime_environment=binding.runtime_environment,
        vector_dimension=binding.vector_dimension,
        video_frame_count=binding.video_frame_count,
        sampling_version=binding.sampling_version,
        resolution=binding.resolution,
    )


def validate_cosmos_model_directory(model: VideoEmbeddingModelSpec) -> Path:
    """Check staged asset presence and configuration values, without verifying artifact revision."""
    model_dir = validate_model_directory(model.runtime_model_dir, model_key=model.key)
    for filename in ("config.json", "processor_config.json"):
        path = model_dir / filename
        if not path.is_file():
            msg = f"Missing {path}. Run: pixi run --as-is model-download --models {model.key}"
            raise FileNotFoundError(msg)
        config = json.loads(path.read_text(encoding="utf-8"))
        expected = {"num_video_frames": model.video_frame_count, "resolution": model.resolution}
        if filename == "config.json":
            expected["embed_dim"] = model.vector_dimension
        if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected.items()):
            msg = f"Staged {model.key}/{filename} differs from the pinned model-card contract"
            raise ValueError(msg)
    return model_dir


def result_field_names(result_key: str, *, model_key: str) -> tuple[str, str]:
    """Return one deterministic versioned vector field and its metadata sibling."""
    if not is_valid_result_key(result_key):
        msg = f"Invalid embedding result key: {result_key!r}"
        raise ValueError(msg)
    binding = resolve_model_binding(model_key)
    embedding_name = f"embedding_video_{result_key}_{binding.field_key}_v{RESULT_VERSION}"
    return embedding_name, f"{embedding_name}_metadata"


def _ownership_metadata(field_set: str, *, digest: str | None = None) -> dict[bytes, bytes]:
    metadata = {OWNER_METADATA: b"multimodal-embedding", FIELD_SET_METADATA: field_set.encode("utf-8")}
    if digest is not None:
        metadata[CONTRACT_DIGEST_METADATA] = digest.encode("utf-8")
    return metadata


def logical_result_fields(result_key: str, *, model_key: str) -> pa.Schema:
    """Return the owned fields before a derived contract digest is attached."""
    embedding_name, metadata_name = result_field_names(result_key, model_key=model_key)
    binding = resolve_model_binding(model_key)
    metadata = pa.struct(
        [
            pa.field("embedding_schema_version", pa.int32(), nullable=False),
            pa.field("result_key", pa.string(), nullable=False),
            pa.field("contract_digest", pa.string(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("episode_id", pa.string(), nullable=False),
            pa.field("materialization_digest", pa.string(), nullable=False),
            pa.field("camera_key", pa.string(), nullable=False),
            pa.field("media_sha256", pa.string(), nullable=False),
            pa.field("model_key", pa.string(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("model_revision", pa.string(), nullable=False),
            pa.field("processor_revision", pa.string(), nullable=False),
            pa.field("sampling_version", pa.string(), nullable=False),
            pa.field("episode_duration_ns", pa.int64(), nullable=False),
            pa.field("media_duration_ns", pa.int64()),
            pa.field("max_episode_duration_s", pa.float64()),
            pa.field("error_code", pa.string()),
            pa.field("error_message", pa.large_string()),
        ]
    )
    return pa.schema(
        [
            pa.field(
                embedding_name,
                pa.list_(pa.float32(), binding.vector_dimension),
                nullable=True,
                metadata=_ownership_metadata(embedding_name),
            ),
            pa.field(metadata_name, metadata, nullable=True, metadata=_ownership_metadata(embedding_name)),
        ]
    )


def normalized_embedding_contract(
    model: VideoEmbeddingModelSpec, *, camera_key: str, result_key: str
) -> dict[str, Any]:
    """Return all result-defining values in a canonical JSON-compatible shape."""
    if not camera_key or camera_key != camera_key.strip():
        msg = "camera_key must be non-empty and cannot have surrounding whitespace"
        raise ValueError(msg)
    return {
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "result_version": RESULT_VERSION,
        "result_key": result_key,
        "camera_key": camera_key,
        "model": {
            "key": model.key,
            "field_key": model.field_key,
            "id": model.model_id,
            "revision": model.revision,
            "precision": model.precision,
        },
        "processor": {
            "revision": model.processor_revision,
            "input": "8_rgb_uint8_frames_btchw",
            "resolution": model.resolution,
            "overrides": {},
        },
        "runtime_environment": model.runtime_environment,
        "sampling": {
            "version": model.sampling_version,
            "frame_count": model.video_frame_count,
            "indices": "numpy.linspace(0, frame_count - 1, 8, dtype=int64)",
            "order": "decoded_presentation_order",
            "short_video": "repeat_indices",
        },
        "media_duration_validation": {
            "episode_duration_comparison": "abs(media_duration_ns-episode_duration_ns)>ceil(2e9/stream.average_rate)",
            "nominal_frame_rate": "positive_video_stream_average_rate",
            "missing_timeline_or_rate": "invalid_timestamps",
        },
        "output": {
            "dimension": model.vector_dimension,
            "dtype": "float32",
            "normalization": NORMALIZATION_VERSION,
            "unit_norm_atol": VECTOR_NORM_TOLERANCE,
            "unit_norm_rtol": 1e-5,
        },
        "fields": [_arrow_field_contract(field) for field in logical_result_fields(result_key, model_key=model.key)],
        "terminal_states": {
            "success": "non_null_unit_vector_and_null_error_fields",
            "error": "null_vector_and_non_empty_error_code_and_message",
            "unsupported_duration": "null_vector_and_error_code_unsupported_duration",
            "item_error_codes": sorted(ITEM_ERROR_CODES),
        },
    }


def embedding_contract_digest(model: VideoEmbeddingModelSpec, *, camera_key: str, result_key: str) -> str:
    """Hash the vector contract; eligibility and runtime settings are intentionally absent."""
    return canonical_digest(normalized_embedding_contract(model, camera_key=camera_key, result_key=result_key))


def embedding_contract_digest_from_config(config: ResolvedMultimodalEmbeddingConfig, *, camera_key: str) -> str:
    """Resolve the vector contract from one public config, ignoring selection and resources."""
    model = resolve_model(config.model.variant)
    return embedding_contract_digest(model, camera_key=camera_key, result_key=config.result_key_for_camera(camera_key))


def result_fields(result_key: str, *, model_key: str, digest: str) -> pa.Schema:
    """Return physical fields, both tagged with the same derived contract digest."""
    logical = logical_result_fields(result_key, model_key=model_key)
    embedding_name, _metadata_name = result_field_names(result_key, model_key=model_key)
    return pa.schema([field.with_metadata(_ownership_metadata(embedding_name, digest=digest)) for field in logical])


def staged_result_schema(result_key: str, *, model_key: str, digest: str) -> pa.Schema:
    """Use a nullable Parquet list for staging; Lance retains the fixed-width vector."""
    fields = result_fields(result_key, model_key=model_key, digest=digest)
    vector_name, metadata_name = result_field_names(result_key, model_key=model_key)
    vector_field = fields.field(vector_name).with_type(pa.list_(pa.float32()))
    return pa.schema([*EPISODE_MAPPED_INPUT_SCHEMA, vector_field, fields.field(metadata_name)])


def validate_field_set(existing: Sequence[pa.Field], *, result_key: str, model_key: str, digest: str) -> None:
    """Refuse a reused result namespace whose exact registered contract differs."""
    expected = result_fields(result_key, model_key=model_key, digest=digest)
    observed = {field.name: field for field in existing}
    missing = [field.name for field in expected if field.name not in observed]
    if missing:
        msg = f"Embedding result field set is incomplete; missing {', '.join(missing)}"
        raise ValueError(msg)
    mismatched = [field.name for field in expected if not observed[field.name].equals(field, check_metadata=True)]
    if mismatched:
        msg = f"Embedding result field set is incompatible for result_key {result_key!r}: {', '.join(mismatched)}"
        raise ValueError(msg)


def build_terminal_result_row(
    source: dict[str, Any],
    *,
    model: VideoEmbeddingModelSpec,
    result_key: str,
    digest: str,
    result: EmbeddingTerminalResult,
) -> dict[str, Any]:
    """Build a field-shaped terminal result row from one projected episode."""
    embedding_name, metadata_name = result_field_names(result_key, model_key=model.key)
    duration_ns = int(source["exclusive_end_ns"]) - int(source["start_ns"])
    row = {
        embedding_name: list(result.vector) if result.vector is not None else None,
        metadata_name: {
            "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
            "result_key": result_key,
            "contract_digest": digest,
            "status": result.status,
            "episode_id": source["episode_id"],
            "materialization_digest": source["materialization_digest"],
            "camera_key": source["camera_key"],
            "media_sha256": source["media_sha256"],
            "model_key": model.key,
            "model_id": model.model_id,
            "model_revision": model.revision,
            "processor_revision": model.processor_revision,
            "sampling_version": model.sampling_version,
            "episode_duration_ns": duration_ns,
            "media_duration_ns": result.media_duration_ns,
            "max_episode_duration_s": result.max_episode_duration_s,
            "error_code": result.error_code,
            "error_message": result.error_message,
        },
    }
    validate_terminal_row(
        row,
        model=model,
        camera_key=str(source["camera_key"]),
        result_key=result_key,
        digest=digest,
    )
    return row


def validate_terminal_row(
    row: dict[str, Any],
    *,
    model: VideoEmbeddingModelSpec,
    camera_key: str,
    result_key: str,
    digest: str,
) -> None:
    """Validate one staged or published result against the selected contract."""
    embedding_name, metadata_name = result_field_names(result_key, model_key=model.key)
    vector = row.get(embedding_name)
    metadata = row.get(metadata_name)
    if not isinstance(metadata, dict):
        msg = "Terminal embedding row requires metadata"
        raise TypeError(msg)
    _validate_metadata_identity(row, metadata, model=model, camera_key=camera_key, result_key=result_key, digest=digest)
    duration_ns, media_duration_ns, max_duration_s = _validate_duration_metadata(row, metadata)
    _validate_terminal_outcome(
        vector,
        metadata,
        model=model,
        duration_ns=duration_ns,
        media_duration_ns=media_duration_ns,
        max_duration_s=max_duration_s,
    )


def validate_terminal_batch(
    table: pa.Table,
    *,
    model: VideoEmbeddingModelSpec,
    camera_key: str,
    result_key: str,
    digest: str,
) -> None:
    """Validate completed Arrow rows with one vectorized finite/norm check."""
    embedding_name, metadata_name = result_field_names(result_key, model_key=model.key)
    vectors = table[embedding_name].combine_chunks()
    metadata_rows = table[metadata_name].to_pylist()
    vector_nulls = vectors.is_null().to_numpy(zero_copy_only=False)
    successful: list[int] = []
    for index, metadata in enumerate(metadata_rows):
        if not isinstance(metadata, dict):
            msg = "Terminal embedding row requires metadata"
            raise TypeError(msg)
        if bool(vector_nulls[index]) != (metadata.get("status") != "success"):
            msg = f"Terminal embedding row {index} has vector nullability incompatible with its status"
            raise ValueError(msg)
        _validate_metadata_identity(
            {}, metadata, model=model, camera_key=camera_key, result_key=result_key, digest=digest
        )
        duration_ns, media_duration_ns, max_duration_s = _validate_duration_metadata({}, metadata)
        if metadata.get("status") == "success":
            if metadata.get("error_code") is not None or metadata.get("error_message") is not None:
                msg = "Successful embedding row cannot carry an error"
                raise ValueError(msg)
            successful.append(index)
        else:
            _validate_terminal_outcome(
                None,
                metadata,
                model=model,
                duration_ns=duration_ns,
                media_duration_ns=media_duration_ns,
                max_duration_s=max_duration_s,
            )
    if not successful:
        return
    values = vectors.values.to_numpy(zero_copy_only=False).reshape(len(vectors), model.vector_dimension)
    selected = values[successful].astype(np.float64, copy=False)
    if not np.isfinite(selected).all():
        msg = "Successful embedding row requires finite vector values"
        raise ValueError(msg)
    norms = np.linalg.norm(selected, axis=1)
    if not np.isclose(norms, 1.0, atol=VECTOR_NORM_TOLERANCE).all():
        msg = f"Successful embedding vector must have unit norm within {VECTOR_NORM_TOLERANCE}"
        raise ValueError(msg)


def _validate_metadata_identity(  # noqa: PLR0913 -- source and contract values are independent inputs
    row: dict[str, Any],
    metadata: dict[str, Any],
    *,
    model: VideoEmbeddingModelSpec,
    camera_key: str,
    result_key: str,
    digest: str,
) -> None:
    expected = {
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "result_key": result_key,
        "contract_digest": digest,
        "camera_key": camera_key,
        "model_key": model.key,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "processor_revision": model.processor_revision,
        "sampling_version": model.sampling_version,
    }
    mismatched = [name for name, value in expected.items() if metadata.get(name) != value]
    if mismatched:
        msg = f"Terminal embedding metadata has incompatible value(s): {', '.join(mismatched)}"
        raise ValueError(msg)
    for name in ("episode_id", "materialization_digest", "media_sha256"):
        if not isinstance(metadata.get(name), str) or not metadata[name]:
            msg = f"Terminal embedding metadata requires non-empty {name}"
            raise ValueError(msg)
        if name in row and row[name] != metadata[name]:
            msg = f"Terminal embedding metadata {name} does not match the source row"
            raise ValueError(msg)
    if "camera_key" in row and row["camera_key"] != metadata["camera_key"]:
        msg = "Terminal embedding metadata camera_key does not match the source row"
        raise ValueError(msg)


def _validate_duration_metadata(row: dict[str, Any], metadata: dict[str, Any]) -> tuple[int, int | None, float | None]:
    duration_ns = metadata.get("episode_duration_ns")
    if not isinstance(duration_ns, int) or isinstance(duration_ns, bool) or duration_ns <= 0:
        msg = "Terminal embedding metadata requires a positive episode_duration_ns"
        raise ValueError(msg)
    if "start_ns" in row and "exclusive_end_ns" in row and row["exclusive_end_ns"] - row["start_ns"] != duration_ns:
        msg = "Terminal embedding metadata episode_duration_ns does not match the source row"
        raise ValueError(msg)
    media_duration_ns = metadata.get("media_duration_ns")
    if media_duration_ns is not None and (
        not isinstance(media_duration_ns, int) or isinstance(media_duration_ns, bool) or media_duration_ns <= 0
    ):
        msg = "Terminal embedding metadata media_duration_ns must be positive or null"
        raise ValueError(msg)
    max_duration_s = metadata.get("max_episode_duration_s")
    if max_duration_s is not None and (
        not isinstance(max_duration_s, (int, float))
        or isinstance(max_duration_s, bool)
        or not math.isfinite(max_duration_s)
        or max_duration_s <= 0
    ):
        msg = "Terminal embedding metadata max_episode_duration_s must be positive and finite or null"
        raise ValueError(msg)
    return duration_ns, media_duration_ns, max_duration_s


def _validate_terminal_outcome(  # noqa: PLR0913 -- validator receives all checked duration facts
    vector: object,
    metadata: dict[str, Any],
    *,
    model: VideoEmbeddingModelSpec,
    duration_ns: int,
    media_duration_ns: int | None,
    max_duration_s: float | None,
) -> None:
    status = metadata.get("status")
    if status not in TERMINAL_STATUSES:
        msg = f"Unsupported embedding terminal status: {status!r}"
        raise ValueError(msg)
    if status == "success":
        _validate_success_vector(vector, model=model)
        if metadata.get("error_code") is not None or metadata.get("error_message") is not None:
            msg = "Successful embedding row cannot carry an error"
            raise ValueError(msg)
    elif vector is not None:
        msg = f"{status} embedding row requires a null vector"
        raise ValueError(msg)
    if status == "error" and (
        not isinstance(metadata.get("error_code"), str)
        or metadata["error_code"] not in ITEM_ERROR_CODES
        or not isinstance(metadata.get("error_message"), str)
        or not metadata["error_message"]
    ):
        msg = "Error embedding row requires a supported error code and non-empty message"
        raise ValueError(msg)
    if status == "error" and metadata["error_code"] == "duration_mismatch" and media_duration_ns is None:
        msg = "Duration-mismatch error requires media_duration_ns"
        raise ValueError(msg)
    if status == "unsupported_duration":
        if metadata.get("error_code") != "unsupported_duration" or metadata.get("error_message") is not None:
            msg = "Unsupported-duration row requires error_code='unsupported_duration' and null error_message"
            raise ValueError(msg)
        if max_duration_s is None or duration_ns <= max_duration_s * 1_000_000_000:
            msg = "Unsupported-duration row requires an episode longer than its recorded maximum"
            raise ValueError(msg)


def _validate_success_vector(vector: object, *, model: VideoEmbeddingModelSpec) -> None:
    if (
        not isinstance(vector, (Sequence, np.ndarray))
        or isinstance(vector, (str, bytes))
        or len(vector) != model.vector_dimension
    ):
        msg = f"Successful embedding row requires a {model.vector_dimension}-value vector"
        raise ValueError(msg)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        for value in vector
    ):
        msg = "Successful embedding row requires finite vector values"
        raise ValueError(msg)
    norm = math.sqrt(math.fsum(float(value) ** 2 for value in vector))
    if not np.isclose(norm, 1.0, atol=VECTOR_NORM_TOLERANCE):
        msg = f"Successful embedding vector must have unit norm within {VECTOR_NORM_TOLERANCE}"
        raise ValueError(msg)


def _arrow_field_contract(field: pa.Field) -> dict[str, Any]:
    if pa.types.is_struct(field.type):
        field_type: dict[str, Any] = {
            "name": "struct",
            "fields": [_arrow_field_contract(child) for child in field.type],
        }
    elif pa.types.is_fixed_size_list(field.type):
        field_type = {
            "name": "fixed_size_list",
            "list_size": field.type.list_size,
            "value_type": str(field.type.value_type),
        }
    else:
        field_type = {"name": str(field.type)}
    metadata = {key.decode("utf-8"): value.decode("utf-8") for key, value in sorted((field.metadata or {}).items())}
    return {"name": field.name, "type": field_type, "nullable": field.nullable, "metadata": metadata}
