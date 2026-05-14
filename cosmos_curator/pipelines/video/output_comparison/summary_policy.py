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
"""Default comparison policy for split pipeline summaries.

Policy data is declarative:
* Which fields are compared exactly
* Which fields are compared with a tolerance
* etc.
"""

import attrs


@attrs.define(frozen=True)
class SummaryComparisonPolicy:
    """Field groups used to build summary comparison rules.

    Attributes:
        exact_top_level_fields: Required top-level summary fields compared with
            exact equality, such as input counts, processed counts, model
            identity, and aggregate clip accounting.
        token_fields: Required top-level token accounting fields compared with
            the configured absolute or relative token-count tolerance instead
            of exact equality.
        common_video_fields: Per-video fields required on every video summary
            variant and compared with exact equality, whether the video was
            processed or unprocessed.
        processed_video_fields: Per-video fields required only on processed
            video summaries and compared with exact equality when both outputs
            processed the same video.
        clip_list_fields: Processed-video fields containing JSON arrays of
            clip UUIDs; these are compared by list length and UUID set
            membership rather than by exact list order.

    """

    exact_top_level_fields: tuple[str, ...]
    token_fields: tuple[str, ...]
    common_video_fields: tuple[str, ...]
    processed_video_fields: tuple[str, ...]
    clip_list_fields: tuple[str, ...]


DEFAULT_SUMMARY_POLICY = SummaryComparisonPolicy(
    exact_top_level_fields=(
        "num_input_videos",
        "num_input_videos_selected",
        "num_processed_videos",
        "embedding_algorithm",
        "total_video_duration",
        "total_clip_duration",
        "max_clip_duration",
        "total_video_bytes",
        "num_remuxed_videos",
        "total_num_clips_filtered_by_motion",
        "total_num_clips_filtered_by_aesthetic",
        "total_num_clips_filtered_by_qwen_classifier",
        "total_num_clips_filtered_by_qwen_semantic",
        "total_num_clips_filtered_by_artificial_text",
        "total_num_clips_passed",
        "total_num_clips_transcoded",
        "total_num_clips_with_embeddings",
        "total_num_clips_with_caption",
        "total_num_caption_windows",
        "total_num_clips_with_webp",
    ),
    token_fields=("total_prompt_tokens", "total_output_tokens"),
    common_video_fields=("source_video",),
    processed_video_fields=(
        "video_uuid",
        "num_clip_chunks",
        "num_total_clips",
        "num_clips_filtered_by_motion",
        "num_clips_filtered_by_aesthetic",
        "num_clips_filtered_by_qwen_classifier",
        "num_clips_filtered_by_qwen_semantic",
        "num_clips_filtered_by_artificial_text",
        "num_clips_passed",
        "num_clips_transcoded",
        "num_clips_with_embeddings",
        "num_clips_with_caption",
        "num_caption_windows",
        "num_clips_with_webp",
    ),
    clip_list_fields=("clips", "filtered_clips"),
)
