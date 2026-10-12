# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcode bindings: span input, the owned clip fields, and the encoding result contract."""

from functools import partial

import lance

from cosmos_curator.next.enrichment.contracts import EnrichmentBindings, EnrichmentSpec
from cosmos_curator.next.recipes.video_cut.contracts import SPAN_SCHEMA
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from cosmos_curator.next.recipes.video_transcode.contracts import (
    CLIP_FIELDS,
    completed_publication,
    validate_terminal_batch,
)


def resolve_bindings(config: ResolvedVideoTranscodeConfig) -> EnrichmentBindings:
    """Resolve one definition of the clip fields, recovery contract, and runtime settings."""
    return EnrichmentBindings(
        uri=config.input.spans_lance_uri,
        spec=EnrichmentSpec(
            id_column="span_id",
            input_fields=SPAN_SCHEMA,
            output_fields=CLIP_FIELDS,
            validate_batch=partial(validate_terminal_batch, config=config),
            validate_input=partial(_validate_input, config=config),
        ),
        staging_root=config.output.staging_root_uri,
        result_contract={
            "transcode_contract_id": config.transcode.contract_id,
            "media_root": config.input.media_root,
        },
        execution=config.execution,
    )


def _validate_input(dataset: lance.LanceDataset, *, config: ResolvedVideoTranscodeConfig) -> None:
    """Reject a table whose downstream receipt names another encoding contract or media destination."""
    record = completed_publication(dataset)
    if record is not None and (
        record.get("contract_id") != config.transcode.contract_id or record.get("media_root") != config.input.media_root
    ):
        msg = "Table already contains a different transcode contract or media destination"
        raise ValueError(msg)
