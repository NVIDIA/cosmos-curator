# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for dataset dimension helpers."""

import pytest

from cosmos_curator.core.utils.dataset.dimensions import (
    AspectRatio,
    AspectRatioBinSpec,
    AspectRatioBinsSpec,
    Dimensions,
    ResolutionAspectRatio,
    ResolutionAspectRatioFrames,
    ResolutionAspectRatioFramesBinsSpec,
    _round_to_nearest_even,
)


def test_round_to_nearest_even_handles_even_and_odd_values() -> None:
    """Ensure rounding hits the closest even integer with ties rounding up."""
    assert _round_to_nearest_even(2) == 2
    assert _round_to_nearest_even(3) == 4
    assert _round_to_nearest_even(5) == 6  # tie goes to upper even
    assert _round_to_nearest_even(8) == 8
    assert _round_to_nearest_even(9) == 10


def test_dimensions_ratio_and_resize_preserve_aspect() -> None:
    """Resizing uses the shorter edge and keeps ratios via even rounding."""
    dims = Dimensions(width=400, height=200)
    assert dims.w_by_h == pytest.approx(2.0)
    resized = dims.resize_by_shortest_dimension(new_short_size=100)
    assert resized == Dimensions(width=200, height=100)

    portrait = Dimensions(width=300, height=500)
    resized_portrait = portrait.resize_by_shortest_dimension(new_short_size=200)
    assert resized_portrait == Dimensions(width=200, height=334)


def test_aspect_ratio_to_and_from_path_strings() -> None:
    """String conversions round-trip even when embedded in longer paths."""
    ratio = AspectRatio(width=16, height=9)
    assert ratio.to_path_string() == "aspect_ratio_16_9"
    parsed = AspectRatio.from_path_string("foo/bar/aspect_ratio_16_9/baz")
    assert parsed == ratio


def test_resolution_aspect_ratio_path_roundtrip() -> None:
    """Resolution/aspect-ratio composite strings parse correctly."""
    ar = AspectRatio(width=4, height=3)
    resolution_ar = ResolutionAspectRatio(aspect_ratio=ar, resolution="1080")
    path = resolution_ar.to_path_string()
    assert path == "resolution_1080/aspect_ratio_4_3"

    parsed = ResolutionAspectRatio.from_path_string(f"/some/{path}")
    assert parsed == resolution_ar


def test_resolution_aspect_ratio_frames_path_roundtrip() -> None:
    """Resolution/aspect-ratio/duration strings parse correctly."""
    ar = AspectRatio(width=1, height=1)
    rar_frames = ResolutionAspectRatioFrames(aspect_ratio=ar, resolution="720", length="30_inf")
    path = rar_frames.to_path_string()
    assert path == "resolution_720/aspect_ratio_1_1/duration_30_inf"

    parsed = ResolutionAspectRatioFrames.from_path_string(path)
    assert parsed == rar_frames


def test_aspect_ratio_bins_spec_requires_contiguous_ranges() -> None:
    """Bins with gaps trigger validation errors."""
    with pytest.raises(ValueError, match="Expected bins to be contiguous"):
        AspectRatioBinsSpec(
            [
                AspectRatioBinSpec(0.0, 0.5, AspectRatio(width=1, height=2)),
                AspectRatioBinSpec(0.6, 1.0, AspectRatio(width=1, height=1)),
            ],
        )


def test_standard_aspect_ratio_bins_find_expected_bin() -> None:
    """Standard bins bucket images based on width/height ratio."""
    bins = AspectRatioBinsSpec.for_standard_image_datasets()
    dims = Dimensions(width=400, height=300)  # ratio 4:3
    ar = bins.find_appropriate_bin(dims)
    assert ar == AspectRatio(width=4, height=3)

    wide_dims = Dimensions(width=10000, height=100)  # ratio > 10
    assert bins.find_appropriate_bin(wide_dims) is None


def test_resolution_frames_bins_select_resolution_and_frame_bucket() -> None:
    """Video bins consider resolution, aspect ratio, and video length."""
    bins = ResolutionAspectRatioFramesBinsSpec.for_standard_video_datasets()
    dims = Dimensions(width=1920, height=1080)
    bucket = bins.find_appropriate_bin(dims, video_length=8)
    assert bucket == ResolutionAspectRatioFrames(AspectRatio(16, 9), "1080", "5_10")

    high_length_bucket = bins.find_appropriate_bin(dims, video_length=50)
    assert high_length_bucket == ResolutionAspectRatioFrames(AspectRatio(16, 9), "1080", "30_inf")


def test_resolution_frames_bins_accept_low_resolution_inputs() -> None:
    """Low resolution videos are now binned into lt_720 category."""
    bins = ResolutionAspectRatioFramesBinsSpec.for_standard_video_datasets()
    low_res_dims = Dimensions(width=640, height=480)
    bucket = bins.find_appropriate_bin(low_res_dims, video_length=15)
    assert bucket == ResolutionAspectRatioFrames(AspectRatio(4, 3), "lt_720", "10_30")


def test_video_bins_handle_various_lengths() -> None:
    """Test various video lengths are binned correctly."""
    bins = ResolutionAspectRatioFramesBinsSpec.for_standard_video_datasets()
    dims = Dimensions(width=1600, height=900)  # 16:9 aspect ratio, min edge 900 so 720 bin

    # Short video (5-10 seconds)
    short_bucket = bins.find_appropriate_bin(dims, video_length=7)
    assert short_bucket == ResolutionAspectRatioFrames(AspectRatio(16, 9), "720", "5_10")

    # Medium video (10-30 seconds)
    medium_bucket = bins.find_appropriate_bin(dims, video_length=20)
    assert medium_bucket == ResolutionAspectRatioFrames(AspectRatio(16, 9), "720", "10_30")

    # Long video (>30 seconds)
    long_bucket = bins.find_appropriate_bin(dims, video_length=100)
    assert long_bucket == ResolutionAspectRatioFrames(AspectRatio(16, 9), "720", "30_inf")
