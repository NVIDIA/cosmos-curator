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
"""Shared fixtures for image pipeline tests."""

import pathlib

import pytest

from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask

# Local test image (Sintel trailer frame, Creative Commons Attribution 3.0)
_FIXTURES_DIR = pathlib.Path(__file__).parent / "data"
_TEST_IMAGE_PATH = _FIXTURES_DIR / "test_image.jpg"


@pytest.fixture(scope="session")
def image_data_dir() -> pathlib.Path:
    """Path to the image test data directory (contains test_image.jpg)."""
    return _FIXTURES_DIR


@pytest.fixture
def sample_image_task(image_data_dir: pathlib.Path) -> ImagePipeTask:
    """Single ImagePipeTask for test_image.jpg (input_image set; encoded_data filled by Load stage)."""
    image_path = image_data_dir / "test_image.jpg"
    image = Image(input_image=image_path, relative_path="test_image.jpg")
    return ImagePipeTask(session_id=str(image_path), image=image)
