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
"""Typed schema for the split pipeline's ``summary.json``.

Pydantic v2 models with the same ``ConfigDict(frozen=True, strict=True,
extra="forbid")`` triple as the rest of split_comparison -- no field coercion,
no silent typos, immutable once constructed.

``OutputSummary`` is the top-level shape. ``videos`` is a ``str -> VideoSummary``
mapping where ``VideoSummary`` is a discriminated union over the ``processed``
boolean. The raw ``summary.json`` interleaves top-level scalar fields with
per-video entries at the same level; :meth:`OutputSummary.from_json` splits
them into the two pydantic slots before validation.

Validation errors are pydantic ``ValidationError``s. The driver catches those
(plus ``OSError`` / ``json.JSONDecodeError`` at the IO layer) and turns them
into ``summary_load_failed`` issues; the comparator never raises out of
``load_summary``.
"""

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Field

type Number = int | float

_MODEL_CONFIG = ConfigDict(frozen=True, strict=True, extra="forbid")


def _coerce_to_tuple(value: Any) -> Any:  # noqa: ANN401
    """Accept JSON arrays (Python lists) for ``tuple[str, ...]`` fields under strict mode."""
    if isinstance(value, list):
        return tuple(value)
    return value


# JSON arrays decode as ``list``; ``strict=True`` would otherwise reject them against
# the ``tuple[str, ...]`` declaration. The cast is lossless and one-directional so the
# downstream model remains immutable.
StringTuple = Annotated[tuple[str, ...], BeforeValidator(_coerce_to_tuple)]


class ProcessedVideoSummary(BaseModel):
    """Per-video summary entry emitted when the split pipeline processed the input.

    Carries the clip accounting fields plus the passed / filtered clip uuid
    tuples consumed by :func:`discover_clips`.
    """

    model_config = _MODEL_CONFIG

    processed: Literal[True] = True
    source_video: str
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
    clips: StringTuple
    filtered_clips: StringTuple


class UnprocessedVideoSummary(BaseModel):
    """Per-video summary entry emitted when the input produced no processed metadata."""

    model_config = _MODEL_CONFIG

    processed: Literal[False]
    source_video: str


VideoSummary = Annotated[
    ProcessedVideoSummary | UnprocessedVideoSummary,
    Discriminator("processed"),
]


# Top-level scalar fields declared on ``OutputSummary``. Used by :meth:`from_json` to
# split raw summary.json keys into scalar-vs-video buckets.
_SCALAR_FIELDS: frozenset[str] = frozenset(
    {
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
        "total_prompt_tokens",
        "total_output_tokens",
    },
)


class OutputSummary(BaseModel):
    """Top-level ``summary.json`` view.

    The raw JSON object mixes scalar totals with per-video sub-objects at the
    same level. :meth:`from_json` is the canonical constructor: it splits the
    raw dict by key, hands per-video sub-objects to the discriminated
    ``VideoSummary`` union, and validates everything together.
    """

    model_config = _MODEL_CONFIG

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
    videos: Mapping[str, VideoSummary] = Field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Self:
        """Build an ``OutputSummary`` from a freshly-loaded ``summary.json`` dict.

        Keys in :data:`_SCALAR_FIELDS` become top-level model fields. Every other
        mapping-valued key is treated as a per-video entry and routed into
        ``videos`` (with the ``processed`` discriminator filled in as ``True``
        if absent -- the split pipeline omits it for the common processed
        case). Non-scalar / non-mapping unknown keys cause ``extra="forbid"`` to
        raise, surfacing schema drift early.
        """
        scalars = {key: value for key, value in data.items() if key in _SCALAR_FIELDS}
        videos: dict[str, Mapping[str, Any]] = {
            key: _with_processed_default(value)
            for key, value in data.items()
            if key not in _SCALAR_FIELDS and isinstance(value, Mapping)
        }
        return cls.model_validate({**scalars, "videos": videos})


def _with_processed_default(video_fields: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fill the discriminator: split pipeline omits ``processed`` for processed videos."""
    if "processed" in video_fields:
        return video_fields
    return {**video_fields, "processed": True}
