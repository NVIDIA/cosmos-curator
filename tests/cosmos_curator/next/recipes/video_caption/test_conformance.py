# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared enrichment recovery behavior, run against clip fragments and caption fields."""

from tests.cosmos_curator.next.enrichment.conformance import EnrichmentConformance


class TestCaptionConformance(EnrichmentConformance):
    """Video-caption meets the shared checkpoint and recovery contract."""
