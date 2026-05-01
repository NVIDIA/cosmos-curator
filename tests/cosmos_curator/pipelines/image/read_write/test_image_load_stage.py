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

"""Tests for ImageLoadStage."""

import pathlib

from PIL import Image as PILImage

from cosmos_curator.pipelines.image.read_write.image_load_stage import ImageLoadStage
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask


class TestImageLoadStage:
    """Tests for ImageLoadStage."""

    def test_load_local_image_succeeds(self, tmp_path: pathlib.Path) -> None:
        """Loading an existing local image fills encoded_data."""
        image_file = tmp_path / "photo.jpg"
        PILImage.new("RGB", (4, 3), (255, 0, 0)).save(image_file)
        payload = image_file.read_bytes()

        image = Image(input_image=image_file, relative_path="photo.jpg")
        task = ImagePipeTask(session_id=str(image_file), image=image)
        stage = ImageLoadStage(
            input_path=str(tmp_path),
            input_s3_profile_name="default",
            verbose=False,
            log_stats=False,
        )
        # Local path does not require stage_setup (client only used for S3)
        result = stage.process_data([task])

        assert result is not None
        assert len(result) == 1
        loaded = result[0].image.encoded_data.resolve()
        assert loaded is not None
        assert loaded.tobytes() == payload
        assert result[0].image.errors == {}
        assert result[0].image.image_data is not None
        assert result[0].image.image_data.frames.shape == (1, 3, 4, 3)
        assert result[0].image.width == 4
        assert result[0].image.height == 3
        assert result[0].image.image_data.align_timestamps_ns.tolist() == [0]
        assert result[0].image.image_data.sensor_timestamps_ns.tolist() == [0]

    def test_load_local_image_with_log_stats(self, tmp_path: pathlib.Path) -> None:
        """When log_stats=True, task.stage_perf is populated."""
        image_file = tmp_path / "a.png"
        PILImage.new("RGB", (2, 2), (0, 255, 0)).save(image_file)

        task = ImagePipeTask(
            session_id=str(image_file),
            image=Image(input_image=image_file, relative_path="a.png"),
        )
        stage = ImageLoadStage(
            input_path=str(tmp_path),
            input_s3_profile_name="default",
            log_stats=True,
        )
        result = stage.process_data([task])

        assert result is not None
        assert len(result) == 1
        assert len(result[0].stage_perf) > 0
        assert result[0].image.encoded_data.resolve() is not None
        assert result[0].image.image_data is not None
        assert result[0].image.image_data.align_timestamps_ns.tolist() == [0]
        assert result[0].image.image_data.sensor_timestamps_ns.tolist() == [0]

    def test_load_stage_uses_collection_level_sensor_timeline(self, tmp_path: pathlib.Path) -> None:
        """Each task should sample against the shared collection timeline, not a one-image sensor."""
        image_a = tmp_path / "a.jpg"
        image_b = tmp_path / "b.jpg"
        PILImage.new("RGB", (3, 2), (255, 0, 0)).save(image_a)
        PILImage.new("RGB", (3, 2), (0, 255, 0)).save(image_b)

        tasks = [
            ImagePipeTask(
                session_id=str(image_a),
                image=Image(input_image=image_a, relative_path="a.jpg"),
            ),
            ImagePipeTask(
                session_id=str(image_b),
                image=Image(input_image=image_b, relative_path="b.jpg"),
            ),
        ]
        stage = ImageLoadStage(
            input_path=str(tmp_path),
            input_s3_profile_name="default",
            verbose=False,
            log_stats=False,
        )

        result = stage.process_data(tasks)

        assert result is not None
        assert [task.image.image_data.align_timestamps_ns.tolist() for task in result] == [[0], [1]]
        assert [task.image.image_data.sensor_timestamps_ns.tolist() for task in result] == [[0], [1]]

    def test_load_missing_file_sets_error(self, tmp_path: pathlib.Path) -> None:
        """Loading a non-existent path sets errors['download'] and returns task."""
        missing = tmp_path / "missing.jpg"
        assert not missing.exists()

        task = ImagePipeTask(
            session_id=str(missing),
            image=Image(input_image=missing, relative_path="missing.jpg"),
        )
        stage = ImageLoadStage(
            input_path=str(tmp_path),
            input_s3_profile_name="default",
        )
        result = stage.process_data([task])

        assert result is not None
        assert len(result) == 1
        assert "download" in result[0].image.errors
        assert result[0].image.encoded_data.resolve() is None or (result[0].image.encoded_data.nbytes == 0)
