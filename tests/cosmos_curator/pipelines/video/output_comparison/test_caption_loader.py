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
"""Tests for caption view construction."""

from pathlib import Path

from cosmos_curator.pipelines.video.output_comparison.caption_loader import caption_view_from_clip_artifacts
from cosmos_curator.pipelines.video.output_comparison.caption_policy import DEFAULT_CAPTION_POLICY
from cosmos_curator.pipelines.video.output_comparison.caption_schema import CaptionWindowRange, ClipCaptionView
from cosmos_curator.pipelines.video.output_comparison.video_artifacts import LoadedClipArtifacts
from cosmos_curator.pipelines.video.output_comparison.video_schema import ClipComparisonSpec


def _clip_spec(tmp_path: Path) -> ClipComparisonSpec:
    return ClipComparisonSpec(
        video_key="video.mp4",
        clip_id="clip-a",
        output_a=str(tmp_path / "output-a"),
        output_b=str(tmp_path / "output-b"),
        in_a=True,
        in_b=True,
    )


def _loaded_clip(
    tmp_path: Path,
    **overrides: object,
) -> LoadedClipArtifacts:
    values: dict[str, object] = {
        "spec": _clip_spec(tmp_path),
        "metadata_a": None,
        "metadata_b": None,
        "metadata_path_a": str(tmp_path / "output-a" / "metas" / "v0" / "clip-a.json"),
        "metadata_path_b": str(tmp_path / "output-b" / "metas" / "v0" / "clip-a.json"),
        "missing_metadata_a": False,
        "missing_metadata_b": False,
        "invalid_metadata_a": None,
        "invalid_metadata_b": None,
    }
    values.update(overrides)
    return LoadedClipArtifacts(**values)


def _window(start_frame: int, end_frame: int, caption: str, *, caption_status: str = "success") -> dict[str, object]:
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "caption_status": caption_status,
        "qwen_caption": caption,
    }


def test_caption_view_from_clip_artifacts_parses_valid_windows(tmp_path: Path) -> None:
    """Caption views expose parsed window ranges for both output sides."""
    view = caption_view_from_clip_artifacts(
        _loaded_clip(
            tmp_path,
            metadata_a={"windows": [_window(0, 30, "caption a")]},
            metadata_b={"windows": [_window(15, 45, "caption b")]},
        ),
        policy=DEFAULT_CAPTION_POLICY,
    )

    assert view == ClipCaptionView(
        video_key="video.mp4",
        clip_id="clip-a",
        in_a=True,
        in_b=True,
        windows_a=frozenset({CaptionWindowRange(0, 30)}),
        windows_b=frozenset({CaptionWindowRange(15, 45)}),
        metadata_path_a=str(tmp_path / "output-a" / "metas" / "v0" / "clip-a.json"),
        metadata_path_b=str(tmp_path / "output-b" / "metas" / "v0" / "clip-a.json"),
        missing_metadata_a=False,
        missing_metadata_b=False,
        invalid_metadata_a=None,
        invalid_metadata_b=None,
    )


def test_caption_view_from_clip_artifacts_preserves_missing_and_invalid_state(tmp_path: Path) -> None:
    """Caption views carry metadata load state without reparsing raw artifacts."""
    view = caption_view_from_clip_artifacts(
        _loaded_clip(
            tmp_path,
            metadata_a=None,
            metadata_b=None,
            missing_metadata_a=True,
            invalid_metadata_b="output-b/metas/v0/clip-a.json: ValueError: invalid metadata",
        ),
        policy=DEFAULT_CAPTION_POLICY,
    )

    assert view.windows_a == frozenset()
    assert view.windows_b == frozenset()
    assert view.in_a is True
    assert view.in_b is True
    assert view.metadata_path_a == str(tmp_path / "output-a" / "metas" / "v0" / "clip-a.json")
    assert view.metadata_path_b == str(tmp_path / "output-b" / "metas" / "v0" / "clip-a.json")
    assert view.missing_metadata_a is True
    assert view.missing_metadata_b is False
    assert view.invalid_metadata_a is None
    assert view.invalid_metadata_b == "output-b/metas/v0/clip-a.json: ValueError: invalid metadata"


def test_caption_view_from_clip_artifacts_handles_no_windows(tmp_path: Path) -> None:
    """Metadata without caption windows produces an empty caption view."""
    view = caption_view_from_clip_artifacts(
        _loaded_clip(tmp_path, metadata_a={"windows": []}, metadata_b={}),
        policy=DEFAULT_CAPTION_POLICY,
    )

    assert view.windows_a == frozenset()
    assert view.windows_b == frozenset()


def test_caption_view_from_clip_artifacts_filters_non_caption_windows(tmp_path: Path) -> None:
    """Caption views contain only windows with frame ranges and accepted caption fields."""
    view = caption_view_from_clip_artifacts(
        _loaded_clip(
            tmp_path,
            metadata_a={
                "windows": [
                    _window(0, 30, "caption"),
                    _window(30, 60, "truncated caption", caption_status="truncated"),
                    _window(60, 90, "raw error text", caption_status="error"),
                    _window(90, 120, "raw blocked text", caption_status="blocked"),
                    {"start_frame": 120, "end_frame": 150, "qwen_caption": "missing status"},
                    {"start_frame": True, "end_frame": 90, "qwen_caption": "bad frame"},
                    "not a window",
                ]
            },
            metadata_b={
                "windows": [
                    _window(90, 120, "caption"),
                    _window(120, 150, "caption"),
                ]
            },
        ),
        policy=DEFAULT_CAPTION_POLICY,
    )

    assert view.windows_a == frozenset({CaptionWindowRange(0, 30), CaptionWindowRange(30, 60)})
    assert view.windows_b == frozenset({CaptionWindowRange(90, 120), CaptionWindowRange(120, 150)})


def test_clip_caption_view_json_round_trip() -> None:
    """Caption views round-trip through JSON-compatible Ray Data rows."""
    view = ClipCaptionView(
        video_key="video.mp4",
        clip_id="clip-a",
        in_a=True,
        in_b=False,
        windows_a=frozenset({CaptionWindowRange(0, 30)}),
        windows_b=frozenset({CaptionWindowRange(15, 45)}),
        metadata_path_a="output-a/metas/v0/clip-a.json",
        metadata_path_b=None,
        missing_metadata_a=False,
        missing_metadata_b=True,
        invalid_metadata_a=None,
        invalid_metadata_b="invalid",
    )

    assert ClipCaptionView.from_json_dict(view.to_json_dict()) == view
