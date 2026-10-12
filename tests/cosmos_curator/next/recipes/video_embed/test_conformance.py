# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared enrichment recovery behavior, run against clip fragments and embedding fields."""

import pytest

from cosmos_curator.next.recipes.video_embed import pipeline
from cosmos_curator.next.recipes.video_embed.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from tests.cosmos_curator.next.enrichment.conformance import EnrichmentConformance, RecipeUnderTest


def _retune(config: ResolvedVideoEmbedConfig) -> ResolvedVideoEmbedConfig:
    execution = config.execution.model_copy(update={"inference_batch_size": 8, "parquet_rows_per_file": 7})
    return config.model_copy(update={"execution": execution})


def _recontract(config: ResolvedVideoEmbedConfig) -> ResolvedVideoEmbedConfig:
    return config.model_copy(update={"model": config.model.model_copy(update={"variant": "448p"})})


class TestEmbedConformance(EnrichmentConformance):
    """Video-embed meets the shared checkpoint and recovery contract."""

    @pytest.fixture
    def recipe(self) -> RecipeUnderTest:
        """Drive the embedding pipeline with its model's bindings; shared publication records provenance."""
        return RecipeUnderTest(
            pipeline=pipeline,
            resolve_bindings=resolve_bindings,
            retune=_retune,
            recontract=_recontract,
        )
