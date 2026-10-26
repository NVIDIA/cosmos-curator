# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding bindings: clip readiness, the model's owned fields, and its result contract."""

from functools import partial

from cosmos_curator.next.enrichment.contracts import EnrichmentBindings, EnrichmentSpec
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import resolve_model_spec, validate_terminal_batch
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA, validate_clip_readiness


def resolve_bindings(config: ResolvedVideoEmbedConfig) -> EnrichmentBindings:
    """Resolve one definition of the model fields, recovery contract, and runtime settings."""
    model = resolve_model_spec(config.model.variant)
    return EnrichmentBindings(
        uri=config.input.clips_lance_uri,
        spec=EnrichmentSpec(
            id_column="clip_id",
            input_fields=CLIP_SCHEMA,
            output_fields=model.fields,
            result_contract=model.contract,
            validate_batch=partial(validate_terminal_batch, spec=model),
            validate_input=partial(validate_clip_readiness, check_unique_ids=False),
        ),
        staging_root=config.output.staging_root_uri,
        execution=config.execution,
    )
