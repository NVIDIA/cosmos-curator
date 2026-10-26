# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution settings shared by checkpointed enrichment recipes."""

from pydantic import Field

from cosmos_curator.next.core.config import CheckpointedExecutionConfig


class EnrichmentExecutionConfig(CheckpointedExecutionConfig):
    """Selection, read, and publication tuning for enrichment, excluded from result identity.

    Recipes subclass this for their ``execution`` section and add their own
    processing settings. Override a default only for a recipe-specific reason.
    """

    max_fragments: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum fragments in a new publication selection; saved selections resume unchanged. "
            "None selects every pending fragment."
        ),
    )
    lance_read_batch_size: int = Field(default=1_000, ge=1, description="Input rows per Lance scanner batch.")
    publication_memory_gib: float = Field(
        default=8.0,
        gt=0,
        allow_inf_nan=False,
        description="Memory reserved per fragment preparation task in GiB; size for one fragment's staged results.",
    )
