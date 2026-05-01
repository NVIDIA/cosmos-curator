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

"""Tests for image pipeline data model."""

import pathlib

import numpy as np
import pytest

from cosmos_curator.core.sensors.data.image_data import ImageData, ImageMetadata
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask
from cosmos_curator.pipelines.video.utils.data_model import TokenCounts


class TestImage:
    """Tests for Image."""

    def test_construction_minimal(self, tmp_path: pathlib.Path) -> None:
        """Image with path only has default encoded_data and errors."""
        p = tmp_path / "photo.jpg"
        img = Image(input_image=p, relative_path="photo.jpg")
        assert img.input_image == p
        assert img.relative_path == "photo.jpg"
        assert img.encoded_data.resolve() is None
        assert img.encoded_data.nbytes == 0
        assert img.errors == {}
        assert img.get_major_size() == 0

    def test_construction_with_encoded_data(self, tmp_path: pathlib.Path) -> None:
        """Image get_major_size reflects encoded_data nbytes."""
        p = tmp_path / "photo.png"
        raw = b"\xff\xd8\xff\xe0\x00\x10"
        arr = bytes_to_numpy(raw)
        img = Image(input_image=p, relative_path="photo.png", encoded_data=arr)
        assert img.encoded_data.nbytes == len(raw)
        assert img.get_major_size() == len(raw)
        resolved = img.encoded_data.resolve()
        assert resolved is not None
        assert resolved.tobytes() == raw

    def test_errors_dict_mutable(self) -> None:
        """Errors dict can be updated by stages."""
        img = Image(input_image=pathlib.Path("/x.jpg"))
        img.errors["download"] = "not found"
        assert img.errors["download"] == "not found"

    def test_image_can_store_image_data(self) -> None:
        """Image can carry decoded sensor-backed image data."""
        image_data = ImageData.from_frames(
            np.zeros((1, 2, 3, 3), dtype=np.uint8),
            ImageMetadata(height=2, width=3, image_format="png"),
        )
        img = Image(input_image=pathlib.Path("/x.png"), image_data=image_data)
        assert img.image_data is image_data

    def test_image_data_rejects_frame_shape_mismatch(self) -> None:
        """ImageData should reject frames whose geometry disagrees with metadata."""
        with pytest.raises(ValueError, match=r"frames must have shape \(N, 2, 3, 3\)"):
            ImageData(
                align_timestamps_ns=np.array([0], dtype=np.int64),
                sensor_timestamps_ns=np.array([0], dtype=np.int64),
                frames=np.zeros((1, 4, 5, 3), dtype=np.uint8),
                metadata=ImageMetadata(height=2, width=3, image_format="png"),
            )

    def test_image_data_rejects_length_mismatch(self) -> None:
        """ImageData should reject arrays whose lengths do not stay aligned."""
        with pytest.raises(ValueError, match="All arrays must be the same length"):
            ImageData(
                align_timestamps_ns=np.array([0, 1], dtype=np.int64),
                sensor_timestamps_ns=np.array([0], dtype=np.int64),
                frames=np.zeros((1, 2, 3, 3), dtype=np.uint8),
                metadata=ImageMetadata(height=2, width=3, image_format="png"),
            )

    def test_image_caption_fields_default_to_video_style_empty_values(self) -> None:
        """Image exposes normalized caption fields compatible with the video pipeline."""
        img = Image(input_image=pathlib.Path("/x.png"))
        assert img.caption == ""
        assert img.captions == {}
        assert img.filter_captions == {}
        assert img.token_counts == {}
        assert img.caption_status is None
        assert img.caption_failure_reason is None
        assert img.filter_caption_status == {}
        assert img.filter_caption_failure_reason == {}
        assert img.qwen_type_classification is None
        assert img.qwen_rejection_stage is None
        assert img.is_filtered is False
        assert img.has_caption() is False

    def test_has_caption_depends_on_caption_status(self) -> None:
        """Only successful/truncated normalized outcomes count as a usable caption."""
        img = Image(input_image=pathlib.Path("/x.png"), caption="caption", token_counts={"qwen": TokenCounts(1, 2)})
        img.caption_status = "success"
        assert img.has_caption() is True
        img.caption_status = "truncated"
        assert img.has_caption() is True
        img.caption_status = "error"
        assert img.has_caption() is False


class TestImagePipeTask:
    """Tests for ImagePipeTask."""

    def test_construction(self) -> None:
        """ImagePipeTask wraps Image and session_id."""
        img = Image(input_image=pathlib.Path("/data/a.jpg"), relative_path="a.jpg")
        task = ImagePipeTask(session_id="/data/a.jpg", image=img)
        assert task.session_id == "/data/a.jpg"
        assert task.image is img
        assert task.weight == 1.0
        assert task.fraction == 1.0
        assert task.get_major_size() == 0

    def test_get_major_size_delegates_to_image(self) -> None:
        """Task get_major_size equals image get_major_size."""
        raw = np.frombuffer(b"xyz", dtype=np.uint8)
        img = Image(
            input_image=pathlib.Path("/b.png"),
            encoded_data=LazyData.coerce(raw),
        )
        task = ImagePipeTask(session_id="/b.png", image=img)
        assert task.get_major_size() == img.get_major_size() == 3
