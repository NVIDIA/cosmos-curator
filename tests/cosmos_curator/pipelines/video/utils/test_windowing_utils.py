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
"""Tests for windowing_utils source-timeline mapping helpers."""

from uuid import uuid4

import pytest

from cosmos_curator.pipelines.video.utils.data_model import Clip, Window
from cosmos_curator.pipelines.video.utils.windowing_utils import (
    estimate_native_frame_count,
    frame_index_to_source_time_s,
    window_source_time_bounds_from_clip,
    window_source_time_bounds_s,
    window_source_time_trace_attributes,
)


def _make_clip(
    span: tuple[float, float] = (10.0, 20.0),
    windows: list[Window] | None = None,
) -> Clip:
    """Build a minimal ``Clip`` with optional pre-attached windows."""
    clip = Clip(uuid=uuid4(), source_video="s3://bucket/video.mp4", span=span)
    if windows:
        clip.windows.extend(windows)
    return clip


# ---------------------------------------------------------------------------
# estimate_native_frame_count
# ---------------------------------------------------------------------------


class TestEstimateNativeFrameCount:
    """``estimate_native_frame_count`` picks the best N from available data."""

    def test_two_partitioned_windows(self) -> None:
        """Contiguous partition 0..9, 10..19 yields N=20."""
        w0 = Window(start_frame=0, end_frame=9)
        w1 = Window(start_frame=10, end_frame=19)
        clip = _make_clip(windows=[w0, w1])
        assert estimate_native_frame_count(clip) == 20

    def test_single_window(self) -> None:
        """One window 0..127 yields N=128."""
        w = Window(start_frame=0, end_frame=127)
        clip = _make_clip(windows=[w])
        assert estimate_native_frame_count(clip) == 128

    def test_empty_windows_with_fallback(self) -> None:
        """No clip.windows — falls back to fallback_window.end_frame + 1."""
        clip = _make_clip()
        w = Window(start_frame=0, end_frame=4)
        assert estimate_native_frame_count(clip, fallback_window=w) == 5

    def test_empty_windows_no_fallback_returns_one(self) -> None:
        """No clip.windows and no fallback — degenerate case returns 1."""
        clip = _make_clip()
        assert estimate_native_frame_count(clip) == 1

    def test_fallback_ignored_when_clip_has_windows(self) -> None:
        """Fallback window is not used when clip.windows is populated."""
        w0 = Window(start_frame=0, end_frame=9)
        clip = _make_clip(windows=[w0])
        fallback = Window(start_frame=0, end_frame=99)
        assert estimate_native_frame_count(clip, fallback_window=fallback) == 10


# ---------------------------------------------------------------------------
# frame_index_to_source_time_s
# ---------------------------------------------------------------------------


class TestFrameIndexToSourceTimeS:
    """``frame_index_to_source_time_s`` maps one frame to a source second."""

    def test_first_frame(self) -> None:
        """Frame 0 maps to clip start."""
        assert frame_index_to_source_time_s((10.0, 20.0), 0, 20) == 10.0

    def test_last_frame(self) -> None:
        """Frame N-1 maps to clip end."""
        assert frame_index_to_source_time_s((10.0, 20.0), 19, 20) == 20.0

    def test_midpoint(self) -> None:
        """Middle frame of 21 frames maps to exact midpoint of 10s span."""
        result = frame_index_to_source_time_s((0.0, 10.0), 10, 21)
        assert result == pytest.approx(5.0)

    def test_single_frame_denominator(self) -> None:
        """N=1 clamps denominator to 1; result equals t0."""
        assert frame_index_to_source_time_s((5.0, 15.0), 0, 1) == 5.0

    def test_zero_length_span(self) -> None:
        """Degenerate span always returns t0 regardless of index."""
        assert frame_index_to_source_time_s((7.0, 7.0), 5, 10) == 7.0


# ---------------------------------------------------------------------------
# window_source_time_bounds_s
# ---------------------------------------------------------------------------


class TestWindowSourceTimeBoundsS:
    """``window_source_time_bounds_s`` linearly maps start/end frames."""

    def test_first_window_of_twenty_frames(self) -> None:
        """Window 0..9 on span (10, 20) with N=20."""
        t0, t1 = window_source_time_bounds_s((10.0, 20.0), 0, 9, 20)
        assert t0 == 10.0
        assert t1 == pytest.approx(10.0 + (9 / 19) * 10.0)

    def test_second_window_of_twenty_frames(self) -> None:
        """Window 10..19 on span (10, 20) with N=20 ends at clip end."""
        t0, t1 = window_source_time_bounds_s((10.0, 20.0), 10, 19, 20)
        assert t0 == pytest.approx(10.0 + (10 / 19) * 10.0)
        assert t1 == pytest.approx(20.0)

    def test_single_native_frame(self) -> None:
        """N=1 maps both bounds to clip start."""
        t0, t1 = window_source_time_bounds_s((5.0, 15.0), 0, 0, 1)
        assert t0 == 5.0
        assert t1 == 5.0

    def test_zero_length_clip_span(self) -> None:
        """Degenerate span yields constant times."""
        t0, t1 = window_source_time_bounds_s((3.0, 3.0), 0, 5, 10)
        assert t0 == 3.0
        assert t1 == 3.0

    def test_full_range_window_covers_entire_span(self) -> None:
        """A single window covering 0..(N-1) maps exactly to clip start/end."""
        t0, t1 = window_source_time_bounds_s((0.0, 60.0), 0, 99, 100)
        assert t0 == pytest.approx(0.0)
        assert t1 == pytest.approx(60.0)


class TestWindowSourceTimeBoundsFromClip:
    """``window_source_time_bounds_from_clip`` combines N estimation with mapping."""

    def test_end_to_end_with_two_windows(self) -> None:
        """First window of a 10s clip (span 10..20, 20 native frames)."""
        w0 = Window(start_frame=0, end_frame=9)
        w1 = Window(start_frame=10, end_frame=19)
        clip = _make_clip(span=(10.0, 20.0), windows=[w0, w1])

        t0, t1 = window_source_time_bounds_from_clip(clip, w0)
        assert t0 == 10.0
        assert t1 == pytest.approx(10.0 + (9 / 19) * 10.0)

        t0_b, t1_b = window_source_time_bounds_from_clip(clip, w1)
        assert t1_b == pytest.approx(20.0)
        # Second window starts where first ended (continuous timeline).
        assert t0_b == pytest.approx(10.0 + (10 / 19) * 10.0)

    def test_fallback_when_clip_windows_empty(self) -> None:
        """Uses the passed window for N estimation when clip.windows is empty."""
        w = Window(start_frame=0, end_frame=4)
        clip = _make_clip(span=(0.0, 5.0))

        t0, t1 = window_source_time_bounds_from_clip(clip, w)
        assert t0 == 0.0
        assert t1 == pytest.approx(5.0)


class TestWindowSourceTimeTraceAttributes:
    """``window_source_time_trace_attributes`` returns OTel-safe dict."""

    def test_returns_all_expected_keys(self) -> None:
        """Dict contains source times, clip span, and human-readable bounds."""
        w0 = Window(start_frame=0, end_frame=9)
        w1 = Window(start_frame=10, end_frame=19)
        clip = _make_clip(span=(10.0, 20.0), windows=[w0, w1])

        attrs = window_source_time_trace_attributes(clip, w0)

        assert set(attrs.keys()) == {
            "window.source_start_s",
            "window.source_end_s",
            "window.clip_span_start_s",
            "window.clip_span_end_s",
            "window.source_bounds",
        }
        assert attrs["window.source_start_s"] == 10.0
        assert attrs["window.source_end_s"] == pytest.approx(10.0 + (9 / 19) * 10.0)
        assert attrs["window.clip_span_start_s"] == 10.0
        assert attrs["window.clip_span_end_s"] == 20.0
        assert isinstance(attrs["window.source_bounds"], str)

    def test_returns_empty_dict_on_bad_clip(self) -> None:
        """Malformed clip data does not raise -- returns empty dict."""
        clip = _make_clip(span=(10.0, 20.0))
        bad_window = Window(start_frame=0, end_frame=-1)
        result = window_source_time_trace_attributes(clip, bad_window)
        assert isinstance(result, dict)
