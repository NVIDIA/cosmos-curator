# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared append recovery behavior, run against chunk outcomes and clip rows."""

from tests.cosmos_curator.next.append.conformance import AppendConformance


class TestRobotActionSplitConformance(AppendConformance):
    """Robot-action-split meets the shared checkpoint and append recovery contract."""
