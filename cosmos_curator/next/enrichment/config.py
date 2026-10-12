# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution settings shared by checkpointed enrichment recipes."""

from pydantic import BaseModel, ConfigDict, Field


class EnrichmentExecutionConfig(BaseModel):
    """Storage, checkpoint, and publication tuning excluded from result identity.

    Recipes subclass this for their ``execution`` section and add their own
    processing settings. Override a default only for a recipe-specific reason.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    storage_profile: str = Field(default="default", min_length=1)
    lance_read_batch_size: int = Field(default=1_000, ge=1, description="Input rows per Lance scanner batch.")
    parquet_rows_per_file: int = Field(
        default=1_000, ge=1, description="Staged rows per Parquet file, which is also the checkpoint group size."
    )
    publication_memory_gib: float = Field(
        default=8.0,
        gt=0,
        allow_inf_nan=False,
        description="Memory reserved per fragment preparation task in GiB; size for one fragment's staged results.",
    )
    commit_attempts: int = Field(default=5, ge=1, description="Attempts for field registration and publication.")
    progress: bool = False
