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

"""Tests for sensor-library ``ImageData``."""

import numpy as np
import pytest

from cosmos_curator.core.sensors.data.image_data import ImageData, ImageMetadata


def test_image_data_from_frames_synthesizes_timestamps() -> None:
    """``from_frames`` should create synthetic int64 timestamps when none are supplied."""
    frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    data = ImageData.from_frames(frames, ImageMetadata(height=3, width=4, image_format="png"))

    np.testing.assert_array_equal(data.align_timestamps_ns, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(data.sensor_timestamps_ns, np.array([0, 1], dtype=np.int64))
    assert data.frames.flags.writeable is False


def test_image_data_rejects_wrong_frame_shape() -> None:
    """``ImageData`` should reject frames whose geometry does not match metadata."""
    with pytest.raises(ValueError, match="frames must have shape"):
        ImageData(
            align_timestamps_ns=np.array([0], dtype=np.int64),
            sensor_timestamps_ns=np.array([0], dtype=np.int64),
            frames=np.zeros((1, 3, 4, 3), dtype=np.uint8),
            metadata=ImageMetadata(height=5, width=4, image_format="png"),
        )


def test_image_data_rejects_non_uint8_frames() -> None:
    """``ImageData`` should require uint8 image arrays."""
    with pytest.raises(ValueError, match="frames must have dtype uint8"):
        ImageData(
            align_timestamps_ns=np.array([0], dtype=np.int64),
            sensor_timestamps_ns=np.array([0], dtype=np.int64),
            frames=np.zeros((1, 3, 4, 3), dtype=np.float32),
            metadata=ImageMetadata(height=3, width=4, image_format="png"),
        )


def test_image_data_rejects_mismatched_array_lengths() -> None:
    """``ImageData`` should reject timestamp/frame arrays with different lengths."""
    with pytest.raises(ValueError, match="All arrays must be the same length"):
        ImageData(
            align_timestamps_ns=np.array([0, 1], dtype=np.int64),
            sensor_timestamps_ns=np.array([0], dtype=np.int64),
            frames=np.zeros((2, 3, 4, 3), dtype=np.uint8),
            metadata=ImageMetadata(height=3, width=4, image_format="png"),
        )


def test_image_data_does_not_mutate_caller_owned_arrays() -> None:
    """ImageData should keep the caller's arrays writeable while exposing read-only views."""
    align_timestamps_ns = np.array([0, 1], dtype=np.int64)
    sensor_timestamps_ns = np.array([0, 1], dtype=np.int64)
    frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)

    data = ImageData(
        align_timestamps_ns=align_timestamps_ns,
        sensor_timestamps_ns=sensor_timestamps_ns,
        frames=frames,
        metadata=ImageMetadata(height=3, width=4, image_format="png"),
    )

    assert align_timestamps_ns.flags.writeable is True
    assert sensor_timestamps_ns.flags.writeable is True
    assert frames.flags.writeable is True
    assert data.align_timestamps_ns.flags.writeable is False
    assert data.sensor_timestamps_ns.flags.writeable is False
    assert data.frames.flags.writeable is False
    assert data.align_timestamps_ns is not align_timestamps_ns
    assert data.sensor_timestamps_ns is not sensor_timestamps_ns
    assert data.frames is not frames
    assert np.shares_memory(data.align_timestamps_ns, align_timestamps_ns)
    assert np.shares_memory(data.sensor_timestamps_ns, sensor_timestamps_ns)
    assert np.shares_memory(data.frames, frames)
