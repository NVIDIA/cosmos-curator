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

"""Tests for ImageWriterStage."""

import json
import pathlib

import numpy as np
import numpy.testing as npt

from cosmos_curator.core.sensors.data.image_data import ImageData, ImageMetadata
from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.pipelines.image.read_write.image_writer_stage import ImageWriterStage
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask
from cosmos_curator.pipelines.video.utils.data_model import TokenCounts


class TestImageWriterStage:
    """Tests for ImageWriterStage."""

    def test_writes_image_and_metadata(self, tmp_path: pathlib.Path) -> None:
        """Writer creates images/{id}{ext} and metas/{id}.json with expected content."""
        payload = b"\x89PNG\r\n\x1a\n"
        image = Image(
            input_image=tmp_path / "source.png",
            relative_path="source.png",
            encoded_data=LazyData.coerce(np.frombuffer(payload, dtype=np.uint8)),
        )
        task = ImagePipeTask(session_id=str(tmp_path / "source.png"), image=image)
        stage = ImageWriterStage(
            output_path=str(tmp_path / "out"),
            output_s3_profile_name="default",
        )
        result = stage.process_data([task])

        assert result is not None
        assert len(result) == 1
        assert result[0].image.errors == {}

        out = tmp_path / "out"
        images_dir = out / "images"
        metas_dir = out / "metas"
        assert images_dir.is_dir()
        assert metas_dir.is_dir()
        image_files = list(images_dir.iterdir())
        meta_files = list(metas_dir.iterdir())
        assert len(image_files) == 1
        assert len(meta_files) == 1
        assert image_files[0].suffix == ".png"
        assert image_files[0].read_bytes() == payload
        meta = json.loads(meta_files[0].read_text())
        assert meta["source_path"] == str(tmp_path / "source.png")
        assert meta["relative_path"] == "source.png"
        assert meta["has_caption"] is False
        assert meta["is_filtered"] is False
        assert meta["align_timestamp_ns"] is None
        assert meta["sensor_timestamp_ns"] is None
        assert meta["caption_status"] is None
        assert meta["caption_failure_reason"] is None
        assert "filter_caption_status" not in meta
        assert "filter_caption_failure_reason" not in meta
        assert "qwen_type_classification" not in meta
        assert "qwen_rejection_stage" not in meta
        assert meta["token_counts"] == {}

    def test_writes_errors_into_metadata(self, tmp_path: pathlib.Path) -> None:
        """Writer should persist image errors so malformed filter outputs appear in metadata."""
        payload = b"\xff\xd8\xff"
        image = Image(
            input_image=tmp_path / "source.jpg",
            relative_path="source.jpg",
            encoded_data=LazyData.coerce(np.frombuffer(payload, dtype=np.uint8)),
            errors={"qwen": "malformed_model_output"},
        )
        task = ImagePipeTask(session_id=str(tmp_path / "source.jpg"), image=image)
        stage = ImageWriterStage(
            output_path=str(tmp_path / "out"),
            output_s3_profile_name="default",
        )

        result = stage.process_data([task])

        assert result is not None
        meta_files = list((tmp_path / "out" / "metas").iterdir())
        meta = json.loads(meta_files[0].read_text())
        assert meta["errors"] == {"qwen": "malformed_model_output"}

    def test_skip_task_without_encoded_data(self, tmp_path: pathlib.Path) -> None:
        """Task with no encoded_data gets errors['write'] and no file written."""
        image = Image(input_image=tmp_path / "missing.jpg", relative_path="missing.jpg")
        task = ImagePipeTask(session_id=str(tmp_path / "missing.jpg"), image=image)
        stage = ImageWriterStage(
            output_path=str(tmp_path / "out"),
            output_s3_profile_name="default",
        )
        result = stage.process_data([task])

        assert result is not None
        assert "write" in result[0].image.errors
        assert not (tmp_path / "out" / "images").exists() or (len(list((tmp_path / "out" / "images").iterdir())) == 0)

    def test_writes_normalized_caption_metadata(self, tmp_path: pathlib.Path) -> None:
        """Writer persists normalized caption outputs in plain JSON."""
        payload = b"\xff\xd8\xff"
        image = Image(
            input_image=tmp_path / "source.jpg",
            relative_path="source.jpg",
            encoded_data=LazyData.coerce(np.frombuffer(payload, dtype=np.uint8)),
            image_data=ImageData(
                align_timestamps_ns=np.array([123], dtype=np.int64),
                sensor_timestamps_ns=np.array([456], dtype=np.int64),
                frames=np.zeros((1, 2, 2, 3), dtype=np.uint8),
                metadata=ImageMetadata(height=2, width=2, image_format="jpg"),
            ),
            caption="a red car",
            captions={"qwen": "a red car"},
            token_counts={"qwen": TokenCounts(prompt_tokens=12, output_tokens=34)},
            caption_status="success",
        )
        task = ImagePipeTask(session_id=str(tmp_path / "source.jpg"), image=image)
        stage = ImageWriterStage(
            output_path=str(tmp_path / "out"),
            output_s3_profile_name="default",
        )

        result = stage.process_data([task])

        assert result is not None
        meta_files = list((tmp_path / "out" / "metas").iterdir())
        meta = json.loads(meta_files[0].read_text())
        assert meta["has_caption"] is True
        assert meta["align_timestamp_ns"] == 123
        assert meta["sensor_timestamp_ns"] == 456
        assert meta["caption"] == "a red car"
        assert meta["caption_status"] == "success"
        assert meta["caption_failure_reason"] is None
        assert meta["is_filtered"] is False
        assert meta["token_counts"] == {"qwen": {"prompt_tokens": 12, "output_tokens": 34}}

    def test_writes_filtered_images_to_separate_folder(self, tmp_path: pathlib.Path) -> None:
        """Filtered images should be written under filtered_images/ while metadata stays in metas/."""
        payload = b"\xff\xd8\xff"
        image = Image(
            input_image=tmp_path / "filtered.jpg",
            relative_path="filtered.jpg",
            encoded_data=LazyData.coerce(np.frombuffer(payload, dtype=np.uint8)),
            is_filtered=True,
            qwen_rejection_stage="semantic",
        )
        task = ImagePipeTask(session_id=str(tmp_path / "filtered.jpg"), image=image)
        stage = ImageWriterStage(
            output_path=str(tmp_path / "out"),
            output_s3_profile_name="default",
        )

        result = stage.process_data([task])

        assert result is not None
        filtered_images = list((tmp_path / "out" / "filtered_images").iterdir())
        assert len(filtered_images) == 1
        assert filtered_images[0].read_bytes() == payload
        meta_files = list((tmp_path / "out" / "metas").iterdir())
        meta = json.loads(meta_files[0].read_text())
        assert meta["is_filtered"] is True
        assert meta["qwen_rejection_stage"] == "semantic"

    def test_writes_embedding_artifacts_and_metadata_keys(self, tmp_path: pathlib.Path) -> None:
        """Writer should persist embeddings under per-model directories and record embedding_keys."""
        payload = b"\xff\xd8\xff"
        internvideo2 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        clip = np.array([4.0, 5.0], dtype=np.float32)
        image = Image(
            input_image=tmp_path / "embedded.jpg",
            relative_path="embedded.jpg",
            encoded_data=LazyData.coerce(np.frombuffer(payload, dtype=np.uint8)),
            embeddings={
                "internvideo2": internvideo2,
                "clip": clip,
            },
        )
        task = ImagePipeTask(session_id=str(tmp_path / "embedded.jpg"), image=image)
        stage = ImageWriterStage(
            output_path=str(tmp_path / "out"),
            output_s3_profile_name="default",
        )

        result = stage.process_data([task])

        assert result is not None
        meta_files = list((tmp_path / "out" / "metas").iterdir())
        meta = json.loads(meta_files[0].read_text())
        assert meta["embedding_keys"] == ["internvideo2", "clip"]

        internvideo2_path = tmp_path / "out" / "embeddings" / "internvideo2"
        clip_path = tmp_path / "out" / "embeddings" / "clip"
        internvideo2_files = list(internvideo2_path.iterdir())
        clip_files = list(clip_path.iterdir())
        assert len(internvideo2_files) == 1
        assert len(clip_files) == 1
        npt.assert_array_equal(np.load(internvideo2_files[0]), internvideo2)
        npt.assert_array_equal(np.load(clip_files[0]), clip)
