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
"""Tests for typed split output summary schemas."""

import pytest

from cosmos_curator.pipelines.video.output_comparison.summary_schema import (
    OutputSummary,
    ProcessedVideoSummary,
    UnprocessedVideoSummary,
)

from .conftest import summary, video_summary


def test_output_summary_from_json_dict_exposes_typed_fields_and_retains_extras() -> None:
    """Known fields are typed while unknown fields remain available for future consumers."""
    output_summary = OutputSummary.from_json_dict(
        summary(
            custom_top_level={"kept": True},
            **{
                "video.mp4": video_summary(num_total_clips=2, clips=["clip-a", 7]) | {"custom_video_field": "kept"},
            },
        )
    )

    assert output_summary.num_input_videos == 1
    assert output_summary.embedding_algorithm == "internvideo2"
    assert output_summary.extra_fields == {"custom_top_level": {"kept": True}}
    assert output_summary.has_field("custom_top_level") is True
    assert output_summary.value("custom_top_level") == {"kept": True}
    video = output_summary.videos["video.mp4"]
    assert isinstance(video, ProcessedVideoSummary)
    assert video.common.key == "video.mp4"
    assert video.processed is True
    assert video.video_uuid == "video-uuid"
    assert video.num_total_clips == 2
    assert video.clips == ("clip-a", "7")
    assert video.common.extra_fields == {"custom_video_field": "kept"}


def test_output_summary_from_json_dict_preserves_source_values_for_malformed_known_fields() -> None:
    """Malformed processed per-video fields fail validation."""
    with pytest.raises(TypeError, match=r"summary\.json field 'num_total_clips' must be an integer"):
        OutputSummary.from_json_dict(
            summary(
                **{
                    "video.mp4": video_summary() | {"num_total_clips": "2"},
                },
            )
        )

    with pytest.raises(TypeError, match=r"summary\.json field 'clips' must be a list"):
        OutputSummary.from_json_dict(
            summary(
                **{
                    "video.mp4": video_summary() | {"clips": "clip-a"},
                },
            )
        )


def test_output_summary_from_json_dict_builds_unprocessed_video_summary() -> None:
    """Unprocessed video entries only require common fields."""
    output_summary = OutputSummary.from_json_dict(
        summary(
            **{
                "video.mp4": {
                    "source_video": "/inputs/video.mp4",
                    "processed": False,
                },
            },
        )
    )

    video = output_summary.videos["video.mp4"]
    assert isinstance(video, UnprocessedVideoSummary)
    assert video.processed is False
    assert video.common.key == "video.mp4"
    assert video.common.source_video == "/inputs/video.mp4"


def test_output_summary_from_json_dict_keeps_processed_metadata_as_extra_field() -> None:
    """Top-level metadata dicts are not treated as videos just because they contain processed."""
    output_summary = OutputSummary.from_json_dict(
        summary(
            metadata={
                "processed": True,
            },
        )
    )

    assert set(output_summary.videos) == {"video.mp4"}
    assert output_summary.extra_fields == {"metadata": {"processed": True}}


def test_output_summary_from_json_dict_requires_top_level_fields() -> None:
    """Current split summaries must include required top-level accounting fields."""
    summary_data = summary()
    del summary_data["num_input_videos"]

    with pytest.raises(ValueError, match=r"summary\.json missing required field 'num_input_videos'"):
        OutputSummary.from_json_dict(summary_data)


def test_output_summary_from_json_dict_requires_top_level_field_types() -> None:
    """Required top-level fields must have the writer's expected JSON type."""
    with pytest.raises(TypeError, match=r"summary\.json field 'num_input_videos' must be an integer"):
        OutputSummary.from_json_dict(summary(num_input_videos=True))

    with pytest.raises(TypeError, match=r"summary\.json field 'total_output_tokens' must be a number"):
        OutputSummary.from_json_dict(summary(total_output_tokens="N/A"))

    with pytest.raises(TypeError, match=r"summary\.json field 'embedding_algorithm' must be a string"):
        OutputSummary.from_json_dict(summary(embedding_algorithm=7))


def test_video_summary_requires_source_video() -> None:
    """Entries that look like video summaries must include source_video."""
    with pytest.raises(ValueError, match=r"summary\.json missing required field 'source_video'"):
        OutputSummary.from_json_dict(summary(**{"video.mp4": {"video_uuid": "video-uuid"}}))


def test_processed_video_summary_requires_processed_fields() -> None:
    """Processed entries must include processed-only accounting fields."""
    with pytest.raises(ValueError, match=r"summary\.json missing required field 'video_uuid'"):
        OutputSummary.from_json_dict(summary(**{"video.mp4": {"source_video": "/inputs/video.mp4"}}))


def test_video_summary_requires_processed_to_be_boolean_when_present() -> None:
    """The current processed marker is either absent or a boolean."""
    with pytest.raises(TypeError, match=r"summary\.json video field 'processed' must be a boolean"):
        OutputSummary.from_json_dict(
            summary(**{"video.mp4": {"source_video": "/inputs/video.mp4", "processed": "yes"}})
        )
