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

"""Shared image prep helpers and defaults for image captioning stages."""

import io
import math
from typing import Any

import numpy as np
from PIL import Image as PILImage

IMAGE_FACTOR = 28
DEFAULT_PREP_MIN_PIXELS = 128 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_PREP_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
_MAX_RATIO = 200


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int = IMAGE_FACTOR,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        msg = f"Invalid image dimensions: {height}x{width}"
        raise ValueError(msg)
    if max(height, width) / min(height, width) > _MAX_RATIO:
        msg = f"absolute aspect ratio must be smaller than {_MAX_RATIO}, got {max(height, width) / min(height, width)}"
        raise ValueError(msg)
    resized_h = max(factor, _round_by_factor(height, factor))
    resized_w = max(factor, _round_by_factor(width, factor))
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = _floor_by_factor(height / beta, factor)
        resized_w = _floor_by_factor(width / beta, factor)
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = _ceil_by_factor(height * beta, factor)
        resized_w = _ceil_by_factor(width * beta, factor)
    return resized_h, resized_w


def image_frame_to_png_bytes(frame: np.ndarray[Any, Any]) -> bytes:
    """Encode one RGB image frame as PNG bytes."""
    pil = PILImage.fromarray(frame, mode="RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_image_endpoint_input(
    image_frame: np.ndarray[Any, Any],
    *,
    min_pixels: int,
    max_pixels: int,
) -> dict[str, Any]:
    """Prepare one decoded frame for endpoint upload."""
    height, width = int(image_frame.shape[0]), int(image_frame.shape[1])
    resized_h, resized_w = _smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    resized_h = max(IMAGE_FACTOR, resized_h)
    resized_w = max(IMAGE_FACTOR, resized_w)
    pil = PILImage.fromarray(image_frame, mode="RGB")
    resized = pil.resize((resized_w, resized_h), resample=PILImage.Resampling.BICUBIC)
    frame_hwc = np.asarray(resized, dtype=np.uint8)
    return {
        "payload_bytes": image_frame_to_png_bytes(frame_hwc),
        "height": int(frame_hwc.shape[0]),
        "width": int(frame_hwc.shape[1]),
    }
