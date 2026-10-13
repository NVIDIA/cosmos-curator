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
    contract_id = config.transcode.contract_id
    media_root = config.input.media_root
    return EnrichmentBindings(
        uri=config.input.spans_lance_uri,
        spec=EnrichmentSpec(
            id_column="span_id",
            input_fields=SPAN_SCHEMA,
            output_fields=CLIP_FIELDS,
            result_contract={"transcode_contract_id": contract_id, "media_root": media_root},
            validate_batch=partial(validate_terminal_batch, contract_id=contract_id, media_root=media_root),
            validate_input=partial(_validate_input, contract_id=contract_id, media_root=media_root),
        ),
        staging_root=config.output.staging_root_uri,
        execution=config.execution,
    )


def _validate_input(dataset: lance.LanceDataset, *, contract_id: str, media_root: str) -> None:
    """Reject a table whose downstream receipt names another encoding contract or media destination."""
    record = completed_publication(dataset)
    if record is not None and (record.get("contract_id") != contract_id or record.get("media_root") != media_root):
        msg = "Table already contains a different transcode contract or media destination"
        raise ValueError(msg)
