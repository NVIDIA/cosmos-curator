# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding bindings: clip readiness, the model's owned fields, and its result contract."""

from functools import partial

import lance

from cosmos_curator.next.enrichment.contracts import EnrichmentBindings, EnrichmentSpec
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import resolve_model_spec, validate_terminal_batch
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA


def resolve_bindings(config: ResolvedVideoEmbedConfig) -> EnrichmentBindings:
    """Resolve one definition of the model fields, recovery contract, and runtime settings."""
    model = resolve_model_spec(config.model.variant)
    return EnrichmentBindings(
        uri=config.input.clips_lance_uri,
        spec=EnrichmentSpec(
            id_column="clip_id",
            input_fields=CLIP_SCHEMA,
            output_fields=model.fields,
            validate_batch=partial(validate_terminal_batch, spec=model),
            validate_input=_validate_input,
        ),
        staging_root=config.output.staging_root_uri,
        result_contract=model.contract,
        execution=config.execution,
    )


def _validate_input(dataset: lance.LanceDataset) -> None:
    """Require every clip to have media in one whole-input query."""
    query = (
        "SELECT COUNT(*) AS n, COUNT(clip_uri) AS uris, COUNT(clip_size_bytes) AS sizes, "
        "MIN(clip_size_bytes) AS smallest FROM dataset"
    )
    stats = next(iter(dataset.sql(query).build().to_batch_records())).to_pylist()[0]
    if any(stats[name] != stats["n"] for name in ("uris", "sizes")) or (stats["n"] and stats["smallest"] <= 0):
        msg = "Clips require non-null URIs and positive byte sizes"
        raise ValueError(msg)
