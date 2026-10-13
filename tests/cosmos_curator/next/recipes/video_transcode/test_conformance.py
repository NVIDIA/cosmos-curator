# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared enrichment recovery behavior, run against span fragments and clip metadata."""

from tests.cosmos_curator.next.enrichment.conformance import EnrichmentConformance


class TestTranscodeConformance(EnrichmentConformance):
    """Video-transcode meets the shared checkpoint and recovery contract."""
