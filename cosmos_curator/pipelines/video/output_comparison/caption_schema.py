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
"""Typed schema for split output caption structure comparison."""

from collections.abc import Mapping
from typing import Self, cast

import attrs

from cosmos_curator.pipelines.video.output_comparison.json_types import JsonDictObject, JsonValue


@attrs.define(frozen=True, order=True)
class CaptionWindowRange:
    """Frame range for a caption window."""

    start_frame: int
    end_frame: int

    @property
    def label(self) -> str:
        """Return the split pipeline's stable window key format."""
        return f"{self.start_frame}_{self.end_frame}"

    def to_json_dict(self) -> JsonDictObject:
        """Convert this window range to a JSON-compatible dictionary."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    @classmethod
    def from_json_dict(cls, row: Mapping[str, JsonValue]) -> Self:
        """Build a window range from a JSON-compatible dictionary."""
        return cls(
            start_frame=_required_int(row, "start_frame"),
            end_frame=_required_int(row, "end_frame"),
        )


@attrs.define(frozen=True)
class ClipCaptionView:
    """Normalized caption evidence for one clip comparison row.

    Attributes:
        video_key: Stable video identifier from the output summaries.
        clip_id: Clip UUID for this caption view.
        in_a: Whether this clip UUID is listed for this video in output A.
        in_b: Whether this clip UUID is listed for this video in output B.
        windows_a: Parsed caption window frame ranges for output A.
        windows_b: Parsed caption window frame ranges for output B.
        metadata_path_a: Expected output A metadata path when the clip exists
            on output A.
        metadata_path_b: Expected output B metadata path when the clip exists
            on output B.
        missing_metadata_a: Whether output A metadata was expected but absent.
        missing_metadata_b: Whether output B metadata was expected but absent.
        invalid_metadata_a: Output A metadata load or parse error, if any.
        invalid_metadata_b: Output B metadata load or parse error, if any.

    """

    video_key: str
    clip_id: str
    in_a: bool
    in_b: bool
    windows_a: frozenset[CaptionWindowRange]
    windows_b: frozenset[CaptionWindowRange]
    metadata_path_a: str | None
    metadata_path_b: str | None
    missing_metadata_a: bool
    missing_metadata_b: bool
    invalid_metadata_a: str | None
    invalid_metadata_b: str | None

    def to_json_dict(self) -> JsonDictObject:
        """Convert this caption view to a JSON-compatible Ray Data row."""
        return {
            "video_key": self.video_key,
            "clip_id": self.clip_id,
            "in_a": self.in_a,
            "in_b": self.in_b,
            "windows_a": _window_json_list(self.windows_a),
            "windows_b": _window_json_list(self.windows_b),
            "metadata_path_a": self.metadata_path_a,
            "metadata_path_b": self.metadata_path_b,
            "missing_metadata_a": self.missing_metadata_a,
            "missing_metadata_b": self.missing_metadata_b,
            "invalid_metadata_a": self.invalid_metadata_a,
            "invalid_metadata_b": self.invalid_metadata_b,
        }

    @classmethod
    def from_json_dict(cls, row: Mapping[str, JsonValue]) -> Self:
        """Build a caption view from a JSON-compatible Ray Data row."""
        return cls(
            video_key=_required_str(row, "video_key"),
            clip_id=_required_str(row, "clip_id"),
            in_a=_required_bool(row, "in_a"),
            in_b=_required_bool(row, "in_b"),
            windows_a=_required_window_set(row, "windows_a"),
            windows_b=_required_window_set(row, "windows_b"),
            metadata_path_a=_optional_str(row, "metadata_path_a"),
            metadata_path_b=_optional_str(row, "metadata_path_b"),
            missing_metadata_a=_required_bool(row, "missing_metadata_a"),
            missing_metadata_b=_required_bool(row, "missing_metadata_b"),
            invalid_metadata_a=_optional_str(row, "invalid_metadata_a"),
            invalid_metadata_b=_optional_str(row, "invalid_metadata_b"),
        )


@attrs.define(frozen=True)
class CaptionComparisonCounts:
    """Caption comparison counters emitted by one or more video comparisons."""

    videos_with_captions_a: int = 0
    videos_with_captions_b: int = 0
    clips_with_captions_a: int = 0
    clips_with_captions_b: int = 0
    caption_windows_a: int = 0
    caption_windows_b: int = 0
    videos_compared: int = 0
    clips_compared: int = 0
    windows_compared: int = 0

    def to_json_dict(self) -> JsonDictObject:
        """Convert counts to a JSON-compatible dictionary."""
        return cast("JsonDictObject", attrs.asdict(self))

    @classmethod
    def from_json_dict(cls, row: Mapping[str, JsonValue]) -> Self:
        """Build counts from a Ray Data row."""
        return cls(
            videos_with_captions_a=_required_int(row, "videos_with_captions_a"),
            videos_with_captions_b=_required_int(row, "videos_with_captions_b"),
            clips_with_captions_a=_required_int(row, "clips_with_captions_a"),
            clips_with_captions_b=_required_int(row, "clips_with_captions_b"),
            caption_windows_a=_required_int(row, "caption_windows_a"),
            caption_windows_b=_required_int(row, "caption_windows_b"),
            videos_compared=_required_int(row, "videos_compared"),
            clips_compared=_required_int(row, "clips_compared"),
            windows_compared=_required_int(row, "windows_compared"),
        )


def _required_int(row: Mapping[str, JsonValue], field: str) -> int:
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, int):
        error_msg = f"caption comparison row field {field!r} must be an integer"
        raise TypeError(error_msg)
    return value


def _required_str(row: Mapping[str, JsonValue], field: str) -> str:
    value = row[field]
    if not isinstance(value, str):
        error_msg = f"caption comparison row field {field!r} must be a string"
        raise TypeError(error_msg)
    return value


def _required_bool(row: Mapping[str, JsonValue], field: str) -> bool:
    value = row[field]
    if not isinstance(value, bool):
        error_msg = f"caption comparison row field {field!r} must be a boolean"
        raise TypeError(error_msg)
    return value


def _optional_str(row: Mapping[str, JsonValue], field: str) -> str | None:
    value = row[field]
    if value is not None and not isinstance(value, str):
        error_msg = f"caption comparison row field {field!r} must be a string or null"
        raise TypeError(error_msg)
    return value


def _required_window_set(row: Mapping[str, JsonValue], field: str) -> frozenset[CaptionWindowRange]:
    value = row[field]
    if not isinstance(value, list):
        error_msg = f"caption comparison row field {field!r} must be a list"
        raise TypeError(error_msg)
    windows: set[CaptionWindowRange] = set()
    for item in value:
        if not isinstance(item, dict):
            error_msg = f"caption comparison row field {field!r} must be a list of objects"
            raise TypeError(error_msg)
        windows.add(CaptionWindowRange.from_json_dict(item))
    return frozenset(windows)


def _window_json_list(windows: frozenset[CaptionWindowRange]) -> list[JsonValue]:
    return [window.to_json_dict() for window in sorted(windows)]
