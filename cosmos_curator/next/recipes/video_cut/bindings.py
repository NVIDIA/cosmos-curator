# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video-cut bindings: discovered sources, staged source plans, appended spans, and the cutting contract."""

from cosmos_curator.next.append.contracts import AppendBindings, AppendSpec
from cosmos_curator.next.recipes.video_cut.config import ResolvedVideoCutConfig
from cosmos_curator.next.recipes.video_cut.contracts import (
    INPUT_SCHEMA,
    SOURCE_SCHEMA,
    SPAN_SCHEMA,
    expand_sources,
    validate_source_results,
)


def resolve_bindings(config: ResolvedVideoCutConfig) -> AppendBindings:
    """Resolve one definition of the span table, source plans, recovery contract, and runtime settings.

    The cutting contract is the result contract, which binds the span table.
    """
    return AppendBindings(
        uri=config.output.spans_lance_uri,
        spec=AppendSpec(
            owner="video-cut",
            id_column="source_id",
            input_fields=INPUT_SCHEMA,
            result_fields=SOURCE_SCHEMA,
            output_fields=SPAN_SCHEMA,
            result_contract=config.split.cutting_contract,
            validate_batch=validate_source_results,
            expand=expand_sources,
            error_column="error_stage",
        ),
        staging_root=config.output.staging_root_uri,
        execution=config.execution,
    )
