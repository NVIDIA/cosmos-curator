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
"""Typed schema for split pipeline summary comparison."""

from collections.abc import Mapping
from typing import Literal, Self

import attrs

from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue

Number = int | float

_OUTPUT_SUMMARY_INTERNAL_FIELDS = frozenset({"present_fields", "field_values", "extra_fields", "videos"})
_VIDEO_SUMMARY_COMMON_INTERNAL_FIELDS = frozenset({"key", "present_fields", "field_values", "extra_fields"})
_PROCESSED_VIDEO_INTERNAL_FIELDS = frozenset({"common"})
_UNPROCESSED_VIDEO_INTERNAL_FIELDS = frozenset({"common"})


class MissingSummaryFieldError(ValueError):
    """Required summary field was absent from ``summary.json``."""

    def __init__(self, message: str, *, field: str) -> None:
        """Initialize the error message and missing field name."""
        super().__init__(message)
        self.field = field


class InvalidSummaryFieldError(TypeError):
    """Summary field was present but had the wrong JSON shape."""

    def __init__(self, message: str, *, field: str) -> None:
        """Initialize the error message and invalid field name."""
        super().__init__(message)
        self.field = field


@attrs.define(frozen=True)
class VideoSummaryCommon:
    """Fields shared by all per-video summary variants."""

    key: str
    source_video: str
    present_fields: frozenset[str]
    field_values: Mapping[str, JsonValue]
    extra_fields: Mapping[str, JsonValue]


@attrs.define(frozen=True)
class ProcessedVideoSummary:
    """Summary entry for an input video with processed metadata.

    The split summary writer emits this shape when the corresponding
    ``processed_videos/<input>.json`` record exists. It contains video identity,
    clip accounting, and clip UUID lists aggregated from processed clip chunks.
    """

    common: VideoSummaryCommon
    video_uuid: str
    num_clip_chunks: int
    num_total_clips: int
    num_clips_filtered_by_motion: int
    num_clips_filtered_by_aesthetic: int
    num_clips_filtered_by_qwen_classifier: int
    num_clips_filtered_by_qwen_semantic: int
    num_clips_filtered_by_artificial_text: int
    num_clips_passed: int
    num_clips_transcoded: int
    num_clips_with_embeddings: int
    num_clips_with_caption: int
    num_caption_windows: int
    num_clips_with_webp: int
    clips: tuple[str, ...]
    filtered_clips: tuple[str, ...]
    processed: Literal[True] = attrs.field(default=True, init=False)


@attrs.define(frozen=True)
class UnprocessedVideoSummary:
    """Summary entry for an input video that did not produce processed metadata.

    The split summary writer emits this smaller shape when the input video has no
    ``processed_videos/<input>.json`` record. It preserves the source path and
    marks ``processed`` false, but processed-only accounting fields are absent.
    A corresponding ``video_errors/<input>_<idx>.json`` artifact may explain the
    failure, but the split summary JSON does not currently include those errors.
    """

    common: VideoSummaryCommon
    processed: Literal[False] = attrs.field(default=False, init=False)


type VideoSummary = ProcessedVideoSummary | UnprocessedVideoSummary


@attrs.define(frozen=True)
class OutputSummary:
    """Typed view over a loaded split pipeline summary."""

    present_fields: frozenset[str]
    field_values: Mapping[str, JsonValue]
    extra_fields: Mapping[str, JsonValue]
    videos: Mapping[str, VideoSummary]
    num_input_videos: int
    num_input_videos_selected: int
    num_processed_videos: int
    embedding_algorithm: str
    total_video_duration: Number
    total_clip_duration: Number
    max_clip_duration: Number
    total_video_bytes: int
    num_remuxed_videos: int
    total_num_clips_filtered_by_motion: int
    total_num_clips_filtered_by_aesthetic: int
    total_num_clips_filtered_by_qwen_classifier: int
    total_num_clips_filtered_by_qwen_semantic: int
    total_num_clips_filtered_by_artificial_text: int
    total_num_clips_passed: int
    total_num_clips_transcoded: int
    total_num_clips_with_embeddings: int
    total_num_clips_with_caption: int
    total_num_caption_windows: int
    total_num_clips_with_webp: int
    total_prompt_tokens: Number
    total_output_tokens: Number

    @classmethod
    def from_json_dict(cls, summary_fields: JsonDictObject) -> Self:
        """Build a typed output summary from loaded ``summary.json`` fields."""
        videos: dict[str, VideoSummary] = {}
        for key, value in summary_fields.items():
            if isinstance(value, dict) and _looks_like_video_summary(value):
                videos[key] = _video_summary_from_json_dict(key, value)

        video_keys = set(videos)
        extra_fields = {
            key: value
            for key, value in summary_fields.items()
            if key not in _known_top_level_fields() and key not in video_keys
        }
        return cls(
            present_fields=frozenset(summary_fields),
            field_values=dict(summary_fields),
            extra_fields=extra_fields,
            videos=videos,
            num_input_videos=_required_int_field(summary_fields, "num_input_videos"),
            num_input_videos_selected=_required_int_field(summary_fields, "num_input_videos_selected"),
            num_processed_videos=_required_int_field(summary_fields, "num_processed_videos"),
            embedding_algorithm=_required_str_field(summary_fields, "embedding_algorithm"),
            total_video_duration=_required_number_field(summary_fields, "total_video_duration"),
            total_clip_duration=_required_number_field(summary_fields, "total_clip_duration"),
            max_clip_duration=_required_number_field(summary_fields, "max_clip_duration"),
            total_video_bytes=_required_int_field(summary_fields, "total_video_bytes"),
            num_remuxed_videos=_required_int_field(summary_fields, "num_remuxed_videos"),
            total_num_clips_filtered_by_motion=_required_int_field(
                summary_fields,
                "total_num_clips_filtered_by_motion",
            ),
            total_num_clips_filtered_by_aesthetic=_required_int_field(
                summary_fields,
                "total_num_clips_filtered_by_aesthetic",
            ),
            total_num_clips_filtered_by_qwen_classifier=_required_int_field(
                summary_fields,
                "total_num_clips_filtered_by_qwen_classifier",
            ),
            total_num_clips_filtered_by_qwen_semantic=_required_int_field(
                summary_fields,
                "total_num_clips_filtered_by_qwen_semantic",
            ),
            total_num_clips_filtered_by_artificial_text=_required_int_field(
                summary_fields,
                "total_num_clips_filtered_by_artificial_text",
            ),
            total_num_clips_passed=_required_int_field(summary_fields, "total_num_clips_passed"),
            total_num_clips_transcoded=_required_int_field(summary_fields, "total_num_clips_transcoded"),
            total_num_clips_with_embeddings=_required_int_field(summary_fields, "total_num_clips_with_embeddings"),
            total_num_clips_with_caption=_required_int_field(summary_fields, "total_num_clips_with_caption"),
            total_num_caption_windows=_required_int_field(summary_fields, "total_num_caption_windows"),
            total_num_clips_with_webp=_required_int_field(summary_fields, "total_num_clips_with_webp"),
            total_prompt_tokens=_required_number_field(summary_fields, "total_prompt_tokens"),
            total_output_tokens=_required_number_field(summary_fields, "total_output_tokens"),
        )

    def has_field(self, field: str) -> bool:
        """Return whether the summary contains ``field``."""
        return field in self.present_fields

    def value(self, field: str) -> JsonValue:
        """Return the source value for ``field``."""
        return self.field_values.get(field)


def _looks_like_video_summary(value: Mapping[str, JsonValue]) -> bool:
    return "video_uuid" in value or "source_video" in value


def _video_summary_from_json_dict(key: str, summary_fields: JsonDictObject) -> VideoSummary:
    extra_fields = {field: value for field, value in summary_fields.items() if field not in _known_video_fields()}
    common = VideoSummaryCommon(
        key=key,
        source_video=_required_str_field(summary_fields, "source_video"),
        present_fields=frozenset(summary_fields),
        field_values=dict(summary_fields),
        extra_fields=extra_fields,
    )
    if not _processed_field(summary_fields):
        return UnprocessedVideoSummary(common=common)
    return ProcessedVideoSummary(
        common=common,
        video_uuid=_required_str_field(summary_fields, "video_uuid"),
        num_clip_chunks=_required_int_field(summary_fields, "num_clip_chunks"),
        num_total_clips=_required_int_field(summary_fields, "num_total_clips"),
        num_clips_filtered_by_motion=_required_int_field(summary_fields, "num_clips_filtered_by_motion"),
        num_clips_filtered_by_aesthetic=_required_int_field(summary_fields, "num_clips_filtered_by_aesthetic"),
        num_clips_filtered_by_qwen_classifier=_required_int_field(
            summary_fields,
            "num_clips_filtered_by_qwen_classifier",
        ),
        num_clips_filtered_by_qwen_semantic=_required_int_field(
            summary_fields,
            "num_clips_filtered_by_qwen_semantic",
        ),
        num_clips_filtered_by_artificial_text=_required_int_field(
            summary_fields,
            "num_clips_filtered_by_artificial_text",
        ),
        num_clips_passed=_required_int_field(summary_fields, "num_clips_passed"),
        num_clips_transcoded=_required_int_field(summary_fields, "num_clips_transcoded"),
        num_clips_with_embeddings=_required_int_field(summary_fields, "num_clips_with_embeddings"),
        num_clips_with_caption=_required_int_field(summary_fields, "num_clips_with_caption"),
        num_caption_windows=_required_int_field(summary_fields, "num_caption_windows"),
        num_clips_with_webp=_required_int_field(summary_fields, "num_clips_with_webp"),
        clips=_required_string_tuple_field(summary_fields, "clips"),
        filtered_clips=_required_string_tuple_field(summary_fields, "filtered_clips"),
    )


def _required_field(field_values: Mapping[str, JsonValue], field: str) -> JsonValue:
    if field not in field_values:
        error_msg = f"summary.json missing required field {field!r}"
        raise MissingSummaryFieldError(error_msg, field=field)
    return field_values[field]


def _required_int_field(field_values: Mapping[str, JsonValue], field: str) -> int:
    value = _required_field(field_values, field)
    if isinstance(value, bool) or not isinstance(value, int):
        error_msg = f"summary.json field {field!r} must be an integer"
        raise InvalidSummaryFieldError(error_msg, field=field)
    return value


def _required_number_field(field_values: Mapping[str, JsonValue], field: str) -> Number:
    value = _required_field(field_values, field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        error_msg = f"summary.json field {field!r} must be a number"
        raise InvalidSummaryFieldError(error_msg, field=field)
    return value


def _required_str_field(field_values: Mapping[str, JsonValue], field: str) -> str:
    value = _required_field(field_values, field)
    if not isinstance(value, str):
        error_msg = f"summary.json field {field!r} must be a string"
        raise InvalidSummaryFieldError(error_msg, field=field)
    return value


def _processed_field(field_values: Mapping[str, JsonValue]) -> bool:
    if "processed" not in field_values:
        return True
    value = field_values["processed"]
    if not isinstance(value, bool):
        error_msg = "summary.json video field 'processed' must be a boolean"
        raise InvalidSummaryFieldError(error_msg, field="processed")
    return value


def _required_string_tuple_field(field_values: Mapping[str, JsonValue], field: str) -> tuple[str, ...]:
    value = _required_field(field_values, field)
    if not isinstance(value, list):
        error_msg = f"summary.json field {field!r} must be a list"
        raise InvalidSummaryFieldError(error_msg, field=field)
    return tuple(str(item) for item in value)


def _known_top_level_fields() -> frozenset[str]:
    return frozenset(
        field.name for field in attrs.fields(OutputSummary) if field.name not in _OUTPUT_SUMMARY_INTERNAL_FIELDS
    )


def _known_video_fields() -> frozenset[str]:
    common_fields = {
        field.name
        for field in attrs.fields(VideoSummaryCommon)
        if field.name not in _VIDEO_SUMMARY_COMMON_INTERNAL_FIELDS
    }
    processed_fields = {
        field.name
        for field in attrs.fields(ProcessedVideoSummary)
        if field.name not in _PROCESSED_VIDEO_INTERNAL_FIELDS
    }
    unprocessed_fields = {
        field.name
        for field in attrs.fields(UnprocessedVideoSummary)
        if field.name not in _UNPROCESSED_VIDEO_INTERNAL_FIELDS
    }
    return frozenset((*common_fields, *processed_fields, *unprocessed_fields))
