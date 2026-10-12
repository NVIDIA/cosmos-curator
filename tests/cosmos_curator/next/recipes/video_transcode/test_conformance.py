# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared enrichment recovery behavior, run against span fragments and clip metadata."""

import pytest

from cosmos_curator.next.recipes.video_transcode import pipeline
from cosmos_curator.next.recipes.video_transcode.bindings import resolve_bindings
from cosmos_curator.next.recipes.video_transcode.config import ResolvedVideoTranscodeConfig
from tests.cosmos_curator.next.enrichment.conformance import EnrichmentConformance, RecipeUnderTest


def _retune(config: ResolvedVideoTranscodeConfig) -> ResolvedVideoTranscodeConfig:
    execution = config.execution.model_copy(update={"parquet_rows_per_file": 7, "transcode_cpus": 8.0})
    return config.model_copy(update={"execution": execution})


def _recontract(config: ResolvedVideoTranscodeConfig) -> ResolvedVideoTranscodeConfig:
    return config.model_copy(update={"transcode": config.transcode.model_copy(update={"video_bitrate": "8M"})})


class TestTranscodeConformance(EnrichmentConformance):
    """Video-transcode meets the shared checkpoint and recovery contract."""

    @pytest.fixture
    def recipe(self) -> RecipeUnderTest:
        """Drive the transcode pipeline with its own bindings and provenance."""
        return RecipeUnderTest(
            pipeline=pipeline,
            resolve_bindings=resolve_bindings,
            retune=_retune,
            recontract=_recontract,
        )
