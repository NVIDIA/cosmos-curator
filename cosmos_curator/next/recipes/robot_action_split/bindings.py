# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot-action-split bindings: discovered chunks, staged chunk outcomes, appended clips, and the split contract."""

from functools import partial

from cosmos_curator.next.append.contracts import AppendBindings, AppendSpec
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from cosmos_curator.next.recipes.robot_action_split.contracts import (
    CHUNK_SCHEMA,
    CLIP_SCHEMA,
    INPUT_SCHEMA,
    expand_chunks,
    validate_chunk_results,
)


def resolve_bindings(config: ResolvedRobotActionSplitConfig) -> AppendBindings:
    """Resolve one definition of the clip table, chunk outcomes, recovery contract, and runtime settings.

    The split contract is the result contract, which binds the clip table.
    """
    return AppendBindings(
        uri=config.output.lance_uri,
        spec=AppendSpec(
            owner="robot-action-split",
            id_column="chunk_id",
            input_fields=INPUT_SCHEMA,
            result_fields=CHUNK_SCHEMA,
            output_fields=CLIP_SCHEMA,
            result_contract=config.split_contract,
            validate_batch=partial(validate_chunk_results, media_root=config.output.media_root),
            expand=expand_chunks,
            error_column="error_stage",
        ),
        staging_root=config.output.staging_root_uri,
        execution=config.execution,
    )
