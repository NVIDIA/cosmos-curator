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
"""Utilities for managing dimensions and aspect ratios of images."""

import re

import attrs

from cosmos_curator.core.utils.misc import grouping

# Constants for aspect ratio string representation
_AR_PREFIX = "aspect_ratio_"
_AR_COMPILED_RE = re.compile(r"aspect_ratio_(\d+)_(\d+)")
_RAR_COMPILED_RE = re.compile(r"resolution_([a-zA-Z0-9_]+)/aspect_ratio_(\d+)_(\d+)")
_RARF_COMPILED_RE = re.compile(r"resolution_(\d+)/aspect_ratio_(\d+)_(\d+)/duration_(.+)")


def _round_to_nearest_even(n: int) -> int:
    """Round the provided number to the nearest even integer.

    Args:
        n (int): Number to round.

    Returns:
        int: Nearest even integer.

    """
    # If the number is already even, return it
    if n % 2 == 0:
        return int(n)

    # Find the nearest even numbers
    floor_even = int(n) // 2 * 2  # Largest even number <= n
    ceil_even = floor_even + 2  # Smallest even number > n

    # Return the even number that n is closest to
    return floor_even if n - floor_even < ceil_even - n else ceil_even


@attrs.define(frozen=True)
class Dimensions:
    """Represents the dimensions (height and width) of an image."""

    width: int
    height: int

    @property
    def w_by_h(self) -> float:
        """Calculate the width-to-height ratio.

        Returns:
            float: Width-to-height ratio.

        """
        return float(self.width) / self.height

    def resize_by_shortest_dimension(self, new_short_size: int) -> "Dimensions":
        """Resize the dimensions while maintaining the aspect ratio, based on a new shortest dimension value.

        Args:
            new_short_size (int): New size for the shorter dimension.

        Returns:
            Dimensions: New resized dimensions.

        """
        assert new_short_size % 2 == 0
        h, w = self.height, self.width
        if h < w:
            return Dimensions(_round_to_nearest_even(round((new_short_size / h) * w)), new_short_size)
        return Dimensions(new_short_size, _round_to_nearest_even(round((new_short_size / w) * h)))


@attrs.define(frozen=True)
class AspectRatio:
    """Represents the aspect ratio of an image or a group of images."""

    width: int
    height: int

    def to_path_string(self) -> str:
        """Convert aspect ratio to a string format suitable for path names.

        Returns:
            str: String representation of the aspect ratio.

        """
        return _AR_PREFIX + f"{self.width}_{self.height}"

    @classmethod
    def from_path_string(cls, path_str: str) -> "AspectRatio":
        """Create an AspectRatio object from its string representation.

        Args:
            path_str (str): String representation of the aspect ratio.

        Returns:
            AspectRatio: Corresponding AspectRatio object.

        """
        match = _AR_COMPILED_RE.search(path_str)
        assert match
        w, h = int(match.group(1)), int(match.group(2))
        return AspectRatio(height=h, width=w)


@attrs.define(frozen=True)
class ResolutionAspectRatio:
    """Represents the aspect ratio of an image or a group of images."""

    aspect_ratio: AspectRatio
    resolution: str

    def to_path_string(self) -> str:
        """Convert aspect ratio to a string format suitable for path names.

        Returns:
            str: String representation of the aspect ratio.

        """
        return f"resolution_{self.resolution}/aspect_ratio_{self.aspect_ratio.width}_{self.aspect_ratio.height}"

    @classmethod
    def from_path_string(cls, path_str: str) -> "ResolutionAspectRatio":
        """Create an AspectRatio object from its string representation.

        Args:
            path_str (str): String representation of the aspect ratio.

        Returns:
            AspectRatio: Corresponding AspectRatio object.

        """
        match = _RAR_COMPILED_RE.search(path_str)
        assert match
        resolution = match.group(1)
        w, h = int(match.group(2)), int(match.group(3))
        aspect_ratio = AspectRatio(height=h, width=w)
        return ResolutionAspectRatio(aspect_ratio, resolution)


@attrs.define(frozen=True)
class ResolutionAspectRatioFrames:
    """Represents the resolution/aspect_ratio/number_of_frames of an video or a group of videos."""

    aspect_ratio: AspectRatio
    resolution: str
    length: str

    def to_path_string(self) -> str:
        """Convert aspect ratio to a string format suitable for path names.

        Returns:
            str: String representation of the aspect ratio.

        """
        path = f"resolution_{self.resolution}/aspect_ratio_{self.aspect_ratio.width}_{self.aspect_ratio.height}"
        return f"{path}/duration_{self.length}"

    @classmethod
    def from_path_string(cls, path_str: str) -> "ResolutionAspectRatioFrames":
        """Create an ResolutionAspectRatioFrames object from its string representation.

        Args:
            path_str (str): String representation of the resolution/aspect_ratio/duration.

        Returns:
            ResolutionAspectRatioFrames: Corresponding ResolutionAspectRatioFrames object.

        """
        match = _RARF_COMPILED_RE.search(path_str)
        assert match
        resolution = match.group(1)
        w, h = int(match.group(2)), int(match.group(3))
        aspect_ratio = AspectRatio(height=h, width=w)
        length = match.group(4)
        return ResolutionAspectRatioFrames(aspect_ratio, resolution, length)


@attrs.define
class AspectRatioBinSpec:
    """Defines a bin specification for categorizing images based on their aspect ratios."""

    min_w_by_h: float  # Minimum width-to-height ratio for the bin
    max_w_by_h: float  # Maximum width-to-height ratio for the bin
    aspect_ratio: AspectRatio  # Associated aspect ratio for the bin


@attrs.define
class AspectRatioBinsSpec:
    """Collection of AspectRatioBinSpec objects, providing utilities to categorize and manage aspect ratios."""

    bins: list[AspectRatioBinSpec]

    def __attrs_post_init__(self) -> None:
        """Post-initialization validation."""
        self._validate_contiguous_bins()

    @classmethod
    def for_standard_image_datasets(cls) -> "AspectRatioBinsSpec":
        """Create a standard set of aspect ratio bins suitable for datasets.

        Returns:
            AspectRatioBinsSpec: Collection of bins.

        """
        out = []

        def append_bin(min_ratio: float, max_ratio: float, ar_width: int, ar_height: int) -> None:
            out.append(AspectRatioBinSpec(min_ratio, max_ratio, AspectRatio(ar_width, ar_height)))

        append_bin(0, 0.65, 9, 16)
        append_bin(0.65, 0.88, 3, 4)
        append_bin(0.88, 1.16, 1, 1)
        append_bin(1.16, 1.55, 4, 3)
        append_bin(1.55, 10, 16, 9)

        return AspectRatioBinsSpec(out)

    def _validate_contiguous_bins(self) -> None:
        """Ensure that the aspect ratio bins are contiguous, i.e., there are no gaps between them.

        Raises:
            ValueError: If any non-contiguous bins are found.

        """
        for first, second in grouping.pairwise(self.bins):
            if first.max_w_by_h != second.min_w_by_h:
                error_msg = f"Expected bins to be contiguous, but got {first} and {second}."
                raise ValueError(error_msg)

    def find_appropriate_bin(self, dimensions: Dimensions) -> AspectRatio | None:
        """Identify the appropriate aspect ratio bin for given image dimensions.

        Args:
            dimensions (Dimensions): Image dimensions to categorize.

        Returns:
            AspectRatio: Appropriate aspect ratio bin for the image.

        Raises:
            ValueError: If no suitable bin is found.

        """
        for bin_spec in self.bins:
            if bin_spec.min_w_by_h < dimensions.w_by_h <= bin_spec.max_w_by_h:
                return bin_spec.aspect_ratio
        return None


@attrs.define
class ResolutionAspectRatioBinSpec:
    """Defines a bin specification for categorizing images based on their resolution/aspect_ratio."""

    min_w_by_h: float  # Minimum width-to-height ratio for the bin
    max_w_by_h: float  # Maximum width-to-height ratio for the bin
    aspect_ratio: AspectRatio  # Aspect ratio
    resolution: str  # Resolution, assuming string format but adjust as needed


@attrs.define
class ResolutionAspectRatioBinsSpec:
    """Collection of AspectRatioBinSpec objects.

    Provides utilities to categorize and manage resolution/aspect_ratio/number_of_frames.
    """

    bins: list[ResolutionAspectRatioBinSpec]

    def __attrs_post_init__(self) -> None:
        """Post-initialization validation."""
        self._validate_contiguous_bins()

    @classmethod
    def for_standard_image_datasets(cls) -> "ResolutionAspectRatioBinsSpec":
        """Create a standard set of aspect ratio bins suitable for datasets.

        Returns:
            ResolutionAspectRatioBinSpec: Collection of bins.

        """
        out = []

        def append_bin(min_ratio: float, max_ratio: float, ar_width: int, ar_height: int, resolution: str) -> None:
            out.append(ResolutionAspectRatioBinSpec(min_ratio, max_ratio, AspectRatio(ar_width, ar_height), resolution))

        append_bin(0, 0.65, 9, 16, "lt_720")
        append_bin(0.65, 0.88, 3, 4, "lt_720")
        append_bin(0.88, 1.16, 1, 1, "lt_720")
        append_bin(1.16, 1.55, 4, 3, "lt_720")
        append_bin(1.55, 10, 16, 9, "lt_720")

        append_bin(0, 0.65, 9, 16, "lt_1080")
        append_bin(0.65, 0.88, 3, 4, "lt_1080")
        append_bin(0.88, 1.16, 1, 1, "lt_1080")
        append_bin(1.16, 1.55, 4, 3, "lt_1080")
        append_bin(1.55, 10, 16, 9, "lt_1080")

        append_bin(0, 0.65, 9, 16, "gt_1080")
        append_bin(0.65, 0.88, 3, 4, "gt_1080")
        append_bin(0.88, 1.16, 1, 1, "gt_1080")
        append_bin(1.16, 1.55, 4, 3, "gt_1080")
        append_bin(1.55, 10, 16, 9, "gt_1080")

        return ResolutionAspectRatioBinsSpec(out)

    def _validate_contiguous_bins(self) -> None:
        """Ensure that the aspect ratio bins are contiguous, i.e., there are no gaps between them.

        Raises:
            ValueError: If any non-contiguous bins are found.

        """

    def find_appropriate_bin(self, dimensions: Dimensions) -> ResolutionAspectRatio | None:
        """Identify the appropriate aspect ratio bin for given video metadata.

        Currently the resolution bin has 3 categories:
            greater than 1080 and great than 720, and less than 720.

        Args:
            dimensions (Dimensions): Video dimensions to categorize.

        Returns:
            ResolutionAspectRatioFrames: Appropriate resolution/aspect_ratio/length/ bin for the image.

        Raises:
            ValueError: If no suitable bin is found.

        """
        min_resolution_1080 = 1080
        min_resolution_720 = 720
        width = dimensions.width
        height = dimensions.height
        w_by_h = dimensions.w_by_h

        if min(height, width) >= min_resolution_1080:
            resolution = "gt_1080"
        elif min(height, width) >= min_resolution_720:
            resolution = "lt_1080"
        else:
            resolution = "lt_720"
        for bin_spec in self.bins:
            is_resolution_match = bin_spec.resolution == resolution
            is_aspect_ratio_match = bin_spec.min_w_by_h < w_by_h <= bin_spec.max_w_by_h
            if is_resolution_match and is_aspect_ratio_match:
                return ResolutionAspectRatio(bin_spec.aspect_ratio, resolution)
        return None


@attrs.define
class ResolutionAspectRatioFramesBinSpec:
    """Defines a bin specification for categorizing images based on their resolution/aspect_ratio/number_of_frames."""

    min_w_by_h: float  # Minimum width by height aspect ratio
    max_w_by_h: float  # Maximum width by height aspect ratio
    aspect_ratio: AspectRatio  # Aspect ratio
    min_length: int  # Minimum video length
    max_length: int  # Maximum video length
    lengths: str  # length '5_10', '10_30', '30_inf'
    resolution: str  # Resolution, assuming string format but adjust as needed


@attrs.define
class ResolutionAspectRatioFramesBinsSpec:
    """Collection of AspectRatioBinSpec objects.

    Provides utilities to categorize and manage resolution/aspect_ratio/number_of_frames.
    """

    bins: list[ResolutionAspectRatioFramesBinSpec]

    def __attrs_post_init__(self) -> None:
        """Post-initialization validation."""
        self._validate_contiguous_bins()

    @classmethod
    def for_standard_video_datasets(cls) -> "ResolutionAspectRatioFramesBinsSpec":
        """Create a standard set of bins suitable for datasets.

        Returns:
            ResolutionAspectRatioFramesBinsSpec: Collection of bins.

        """
        out = []

        def append_bin(  # noqa: PLR0913
            min_ratio: float,
            max_ratio: float,
            ar_width: int,
            ar_height: int,
            min_length: int,
            max_length: int,
            lengths: str,
            resolution: str,
        ) -> None:
            out.append(
                ResolutionAspectRatioFramesBinSpec(
                    min_ratio,
                    max_ratio,
                    AspectRatio(ar_width, ar_height),
                    min_length,
                    max_length,
                    lengths,
                    resolution,
                )
            )

        append_bin(0, 0.65, 9, 16, 5, 10, "5_10", "1080")
        append_bin(0.65, 0.88, 3, 4, 5, 10, "5_10", "1080")
        append_bin(0.88, 1.16, 1, 1, 5, 10, "5_10", "1080")
        append_bin(1.16, 1.55, 4, 3, 5, 10, "5_10", "1080")
        append_bin(1.55, 10, 16, 9, 5, 10, "5_10", "1080")
        append_bin(0, 0.65, 9, 16, 5, 10, "5_10", "720")
        append_bin(0.65, 0.88, 3, 4, 5, 10, "5_10", "720")
        append_bin(0.88, 1.16, 1, 1, 5, 10, "5_10", "720")
        append_bin(1.16, 1.55, 4, 3, 5, 10, "5_10", "720")
        append_bin(1.55, 10, 16, 9, 5, 10, "5_10", "720")
        append_bin(0, 0.65, 9, 16, 10, 30, "10_30", "1080")
        append_bin(0.65, 0.88, 3, 4, 10, 30, "10_30", "1080")
        append_bin(0.88, 1.16, 1, 1, 10, 30, "10_30", "1080")
        append_bin(1.16, 1.55, 4, 3, 10, 30, "10_30", "1080")
        append_bin(1.55, 10, 16, 9, 10, 30, "10_30", "1080")
        append_bin(0, 0.65, 9, 16, 10, 30, "10_30", "720")
        append_bin(0.65, 0.88, 3, 4, 10, 30, "10_30", "720")
        append_bin(0.88, 1.16, 1, 1, 10, 30, "10_30", "720")
        append_bin(1.16, 1.55, 4, 3, 10, 30, "10_30", "720")
        append_bin(1.55, 10, 16, 9, 10, 30, "10_30", "720")
        append_bin(0, 0.65, 9, 16, 30, 2**63 - 1, "30_inf", "1080")
        append_bin(0.65, 0.88, 3, 4, 30, 2**63 - 1, "30_inf", "1080")
        append_bin(0.88, 1.16, 1, 1, 30, 2**63 - 1, "30_inf", "1080")
        append_bin(1.16, 1.55, 4, 3, 30, 2**63 - 1, "30_inf", "1080")
        append_bin(1.55, 10, 16, 9, 30, 2**63 - 1, "30_inf", "1080")
        append_bin(0, 0.65, 9, 16, 30, 2**63 - 1, "30_inf", "720")
        append_bin(0.65, 0.88, 3, 4, 30, 2**63 - 1, "30_inf", "720")
        append_bin(0.88, 1.16, 1, 1, 30, 2**63 - 1, "30_inf", "720")
        append_bin(1.16, 1.55, 4, 3, 30, 2**63 - 1, "30_inf", "720")
        append_bin(1.55, 10, 16, 9, 30, 2**63 - 1, "30_inf", "720")

        # Add low resolution bins
        append_bin(0, 0.65, 9, 16, 5, 10, "5_10", "lt_720")
        append_bin(0.65, 0.88, 3, 4, 5, 10, "5_10", "lt_720")
        append_bin(0.88, 1.16, 1, 1, 5, 10, "5_10", "lt_720")
        append_bin(1.16, 1.55, 4, 3, 5, 10, "5_10", "lt_720")
        append_bin(1.55, 10, 16, 9, 5, 10, "5_10", "lt_720")
        append_bin(0, 0.65, 9, 16, 10, 30, "10_30", "lt_720")
        append_bin(0.65, 0.88, 3, 4, 10, 30, "10_30", "lt_720")
        append_bin(0.88, 1.16, 1, 1, 10, 30, "10_30", "lt_720")
        append_bin(1.16, 1.55, 4, 3, 10, 30, "10_30", "lt_720")
        append_bin(1.55, 10, 16, 9, 10, 30, "10_30", "lt_720")
        append_bin(0, 0.65, 9, 16, 30, 2**63 - 1, "30_inf", "lt_720")
        append_bin(0.65, 0.88, 3, 4, 30, 2**63 - 1, "30_inf", "lt_720")
        append_bin(0.88, 1.16, 1, 1, 30, 2**63 - 1, "30_inf", "lt_720")
        append_bin(1.16, 1.55, 4, 3, 30, 2**63 - 1, "30_inf", "lt_720")
        append_bin(1.55, 10, 16, 9, 30, 2**63 - 1, "30_inf", "lt_720")

        return ResolutionAspectRatioFramesBinsSpec(out)

    def _validate_contiguous_bins(self) -> None:
        """Ensure that the aspect ratio bins are contiguous, i.e., there are no gaps between them.

        Raises:
            ValueError: If any non-contiguous bins are found.

        """

    def find_appropriate_bin(self, dimensions: Dimensions, video_length: float) -> ResolutionAspectRatioFrames | None:
        """Identify the appropriate aspect ratio bin for given video metadata.

        Currently the resolution bin has 3 categories:
            greater than 1080 and great than 720, and less than 720.

        Args:
            dimensions (Dimensions): Video dimensions to categorize.
            video_length (float): Video length to categorize.

        Returns:
            ResolutionAspectRatioFrames: Appropriate resolution/aspect_ratio/length/ bin for the image.

        Raises:
            ValueError: If no suitable bin is found.

        """
        min_resolution_1080 = 1080
        min_resolution_720 = 720
        width = dimensions.width
        height = dimensions.height
        w_by_h = dimensions.w_by_h

        if min(height, width) >= min_resolution_1080:
            resolution = "1080"
        elif min(height, width) >= min_resolution_720:
            resolution = "720"
        else:
            resolution = "lt_720"
        for bin_spec in self.bins:
            is_resolution_match = bin_spec.resolution == resolution
            is_aspect_ratio_match = bin_spec.min_w_by_h < w_by_h <= bin_spec.max_w_by_h
            is_length_match = bin_spec.min_length <= video_length <= bin_spec.max_length
            if is_resolution_match and is_aspect_ratio_match and is_length_match:
                return ResolutionAspectRatioFrames(bin_spec.aspect_ratio, resolution, bin_spec.lengths)
        return None
