# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned model-card behavior, vector schemas, and terminal result validation."""

import json
from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from cosmos_curator.core.utils import environment
from cosmos_curator.next.recipes.video_embed.config import ModelVariant
from cosmos_curator.next.utils.identity import canonical_digest

NUM_FRAMES = 8
SCHEMA_VERSION = 1
ADDRESS_SCHEMA = pa.schema([("fragment_id", pa.int64()), ("row_offset", pa.int64()), ("clip_id", pa.string())])


@dataclass(frozen=True)
class ModelSpec:
    """The selected immutable checkpoint and its private, versioned field set."""

    variant: ModelVariant
    model_id: str
    revision: str
    resolution: int
    dimension: int

    @property
    def model_dir(self) -> Path:
        """Use the same pre-staged path on every runtime node."""
        return environment.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR / self.model_id

    @property
    def vector_field(self) -> str:
        """Name a vector without conflating models or preprocessing revisions."""
        return f"embedding_cosmos_embed1_{self.variant}_v{SCHEMA_VERSION}"

    @property
    def metadata_field(self) -> str:
        """Keep terminal status and provenance beside its vector."""
        return f"{self.vector_field}_metadata"

    @cached_property
    def fields(self) -> pa.Schema:
        """Canonical Lance fields, registered together and nullable while pending."""
        ownership = {b"cosmos_curator.owner": b"video-embed", b"cosmos_curator.field_set": self.vector_field.encode()}
        metadata = pa.struct(
            [
                pa.field("embedding_schema_version", pa.int32(), nullable=False),
                pa.field("contract_digest", pa.string(), nullable=False),
                pa.field("model_id", pa.string(), nullable=False),
                pa.field("model_revision", pa.string(), nullable=False),
                pa.field("status", pa.string(), nullable=False),
                pa.field("error_type", pa.string()),
                pa.field("error_message", pa.large_string()),
            ]
        )
        return pa.schema(
            [
                pa.field(self.vector_field, pa.list_(pa.float32(), self.dimension), metadata=ownership),
                pa.field(self.metadata_field, metadata, metadata=ownership),
            ]
        )

    @cached_property
    def contract(self) -> dict[str, Any]:
        """Result-defining choices only; changing these requires a new field version."""
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_revision": self.revision,
            "precision": "bfloat16",
            "sampling": {"frames": NUM_FRAMES, "indices": "numpy.linspace(0, frame_count - 1, 8, dtype=int)"},
            "decode": {"backend": "pynvvideocodec", "order": "presentation", "format": "rgb24", "device": "cuda"},
            "processor": {
                "source": "pinned_checkpoint",
                "overrides": {},
                "resolution": self.resolution,
                "device": "cuda",
            },
            "output": {"source": "visual_proj", "dtype": "float32", "dimension": self.dimension},
            "failure_policy": {
                "object_or_detected_corrupt_media": "terminal_error",
                "shared_or_model_or_decoder": "fail_attempt",
            },
        }

    @cached_property
    def digest(self) -> str:
        """Name one embedding computation independently of batching and placement."""
        return canonical_digest(self.contract)

    @cached_property
    def result_schema(self) -> pa.Schema:
        """Durable Parquet rows retain identities and compact values only.

        Arrow 25 cannot round-trip null fixed-size lists through Parquet. Stage
        ordinary lists and validate/cast their width at the Lance boundary.
        """
        return pa.schema(
            [
                *ADDRESS_SCHEMA,
                # Parquet uses the child name "element"; match it so strict
                # field-metadata validation also survives a durable round trip.
                self.fields.field(self.vector_field).with_type(pa.list_(pa.field("element", pa.float32()))),
                self.fields.field(self.metadata_field),
            ]
        )


def resolve_model_spec(variant: ModelVariant) -> ModelSpec:
    """Resolve all three model-card variants from the repository's pinned registry."""
    registry = json.loads(files("cosmos_curator").joinpath("configs/all_models.json").read_text(encoding="utf-8"))
    record = registry[f"cosmos_embed1_{variant}"]
    return ModelSpec(
        variant=variant,
        model_id=record["model_id"],
        revision=record["version"],
        resolution=int(variant.removesuffix("p")),
        dimension=256 if variant == "224p" else 768,
    )


def validate_model_directory(spec: ModelSpec) -> Path:
    """Require staged weights and the model-card configuration; never download at runtime."""
    for filename in ("config.json", "processor_config.json"):
        config = json.loads((spec.model_dir / filename).read_text(encoding="utf-8"))
        expected = {"num_video_frames": NUM_FRAMES, "resolution": spec.resolution}
        if filename == "config.json":
            expected["embed_dim"] = spec.dimension
        if any(config.get(name) != value for name, value in expected.items()):
            msg = f"Staged {spec.model_id}/{filename} differs from the pinned model-card contract"
            raise ValueError(msg)
    if not any(spec.model_dir.glob("*.safetensors")):
        msg = f"Cosmos-Embed1 weights must be staged at {spec.model_dir}"
        raise FileNotFoundError(msg)
    return spec.model_dir


def terminal_row(row: dict[str, Any], spec: ModelSpec, vector: list[float] | None) -> dict[str, Any]:
    """Attach result provenance without carrying clip bytes or prepared frames."""
    metadata = {
        "embedding_schema_version": SCHEMA_VERSION,
        "contract_digest": spec.digest,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "status": "error" if row.get("error_type") else "success",
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
    }
    result = {name: row[name] for name in ADDRESS_SCHEMA.names}
    result.update({spec.vector_field: vector, spec.metadata_field: metadata})
    validate_terminal_row(result, spec)
    return result


def validate_terminal_row(row: dict[str, Any], spec: ModelSpec) -> None:
    """Reject partial, incompatible, nonfinite, and malformed terminal values."""
    metadata = row[spec.metadata_field]
    expected = {
        "embedding_schema_version": SCHEMA_VERSION,
        "contract_digest": spec.digest,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
    }
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        msg = "Incompatible or missing embedding provenance"
        raise ValueError(msg)
    vector = row[spec.vector_field]
    if metadata.get("status") == "error":
        if vector is not None or any(
            not isinstance(metadata.get(key), str) or not metadata[key] for key in ("error_type", "error_message")
        ):
            msg = "Error embeddings require a null vector and non-empty error details"
            raise ValueError(msg)
        return
    values = np.asarray(vector, dtype=np.float32)
    if (
        metadata.get("status") != "success"
        or metadata.get("error_type") is not None
        or metadata.get("error_message") is not None
        or values.shape != (spec.dimension,)
        or not np.isfinite(values).all()
        # visual_proj is already normalized by the model in BF16. Preserve that
        # output rather than silently introducing another normalization step.
        or not np.isclose(np.linalg.norm(values), 1.0, atol=0.02)
    ):
        msg = "Successful embeddings require a finite unit vector of the model's dimension and no error"
        raise ValueError(msg)
