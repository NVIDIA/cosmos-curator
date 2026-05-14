# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for split output summary comparison policy."""

import attrs

from cosmos_curator.pipelines.video.output_comparison.summary_policy import DEFAULT_SUMMARY_POLICY
from cosmos_curator.pipelines.video.output_comparison.summary_schema import (
    OutputSummary,
    ProcessedVideoSummary,
    VideoSummaryCommon,
)


def _attrs_field_names(cls: type, *, exclude: frozenset[str]) -> set[str]:
    return {field.name for field in attrs.fields(cls) if field.name not in exclude}


def test_default_summary_policy_matches_current_schema_fields() -> None:
    """Default policy fields should drift only when the schema changes deliberately."""
    output_summary_fields = _attrs_field_names(
        OutputSummary,
        exclude=frozenset({"present_fields", "field_values", "extra_fields", "videos"}),
    )
    assert set(DEFAULT_SUMMARY_POLICY.exact_top_level_fields).isdisjoint(DEFAULT_SUMMARY_POLICY.token_fields)
    assert (
        set(DEFAULT_SUMMARY_POLICY.exact_top_level_fields) | set(DEFAULT_SUMMARY_POLICY.token_fields)
        == output_summary_fields
    )

    common_video_fields = _attrs_field_names(
        VideoSummaryCommon,
        exclude=frozenset({"key", "present_fields", "field_values", "extra_fields"}),
    )
    assert set(DEFAULT_SUMMARY_POLICY.common_video_fields) == common_video_fields

    processed_video_fields = _attrs_field_names(
        ProcessedVideoSummary,
        exclude=frozenset({"common", "processed"}),
    )
    assert set(DEFAULT_SUMMARY_POLICY.processed_video_fields).isdisjoint(DEFAULT_SUMMARY_POLICY.clip_list_fields)
    assert (
        set(DEFAULT_SUMMARY_POLICY.processed_video_fields) | set(DEFAULT_SUMMARY_POLICY.clip_list_fields)
        == processed_video_fields
    )
