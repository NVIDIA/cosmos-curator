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
"""Typed schema for video- and clip-level output comparison work."""

from collections.abc import Mapping
from typing import Self

import attrs

from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue


@attrs.define(frozen=True)
class VideoComparisonSpec:
    """One video-level unit of output comparison work.

    Attributes:
        video_key: Stable video identifier from the output summaries.
        output_a: Path or URI string for the first output root.
        output_b: Path or URI string for the second output root.
        clips_a: Clip UUIDs listed for this video in output A's processed
            summary.
        clips_b: Clip UUIDs listed for this video in output B's processed
            summary.

    """

    video_key: str
    output_a: str
    output_b: str
    clips_a: tuple[str, ...]
    clips_b: tuple[str, ...]


@attrs.define(frozen=True)
class ClipComparisonSpec:
    """One clip-level unit of output comparison artifact work.

    Attributes:
        video_key: Stable video identifier from the output summaries.
        clip_id: Clip UUID to load or validate.
        output_a: Path or URI string for the first output root.
        output_b: Path or URI string for the second output root.
        in_a: Whether this clip UUID is listed for this video in output A.
        in_b: Whether this clip UUID is listed for this video in output B.

    """

    video_key: str
    clip_id: str
    output_a: str
    output_b: str
    in_a: bool
    in_b: bool

    def to_json_dict(self) -> JsonDictObject:
        """Convert this spec to a Ray Data row."""
        return {
            "video_key": self.video_key,
            "clip_id": self.clip_id,
            "output_a": self.output_a,
            "output_b": self.output_b,
            "in_a": self.in_a,
            "in_b": self.in_b,
        }

    @classmethod
    def from_json_dict(cls, row: Mapping[str, JsonValue]) -> Self:
        """Build a spec from a Ray Data row."""
        return cls(
            video_key=_required_str(row, "video_key"),
            clip_id=_required_str(row, "clip_id"),
            output_a=_required_str(row, "output_a"),
            output_b=_required_str(row, "output_b"),
            in_a=_required_bool(row, "in_a"),
            in_b=_required_bool(row, "in_b"),
        )


def _required_bool(row: Mapping[str, JsonValue], field: str) -> bool:
    value = row[field]
    if not isinstance(value, bool):
        error_msg = f"video comparison row field {field!r} must be a boolean"
        raise TypeError(error_msg)
    return value


def _required_str(row: Mapping[str, JsonValue], field: str) -> str:
    value = row[field]
    if not isinstance(value, str):
        error_msg = f"video comparison row field {field!r} must be a string"
        raise TypeError(error_msg)
    return value
