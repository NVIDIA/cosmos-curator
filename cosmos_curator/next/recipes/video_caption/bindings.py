# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caption bindings: clip readiness, the model's owned caption fields, and its result contract."""

from functools import partial

from cosmos_curator.next.enrichment.contracts import EnrichmentBindings, EnrichmentSpec
from cosmos_curator.next.recipes.video_caption.config import ResolvedVideoCaptionConfig
from cosmos_curator.next.recipes.video_caption.contracts import (
    caption_contract_digest,
    normalized_caption_contract,
    resolve_model_spec,
    validate_terminal_batch,
)
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA, validate_clip_readiness


def resolve_bindings(config: ResolvedVideoCaptionConfig) -> EnrichmentBindings:
    """Resolve one definition of the caption fields, recovery contract, and runtime settings."""
    spec = resolve_model_spec(config.model.variant)
    max_model_len = config.execution.max_model_len
    return EnrichmentBindings(
        uri=config.input.clips_lance_uri,
        spec=EnrichmentSpec(
            id_column="clip_id",
            input_fields=CLIP_SCHEMA,
            output_fields=spec.fields,
            result_contract=normalized_caption_contract(spec, max_model_len=max_model_len),
            validate_batch=partial(
                validate_terminal_batch, spec=spec, digest=caption_contract_digest(spec, max_model_len=max_model_len)
            ),
            validate_input=partial(validate_clip_readiness, check_unique_ids=False),
        ),
        staging_root=config.output.staging_root_uri,
        execution=config.execution,
    )
