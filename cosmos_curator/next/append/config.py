# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution settings shared by checkpointed append recipes."""

from pydantic import Field

from cosmos_curator.next.core.config import CheckpointedExecutionConfig


class AppendExecutionConfig(CheckpointedExecutionConfig):
    """Selection and fragment tuning for appends, excluded from result identity.

    Recipes subclass this for their ``execution`` section and add their own
    processing settings. Override a default only for a recipe-specific reason.
    """

    max_items: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum work items in a new publication selection; saved selections resume unchanged. None selects "
            "every pending item. A limit publishes in smaller selections and bounds the driver's per-item read "
            "tasks and workspace manifest."
        ),
    )
    rows_per_fragment: int = Field(default=100_000, ge=1, description="Appended rows per Lance fragment.")
