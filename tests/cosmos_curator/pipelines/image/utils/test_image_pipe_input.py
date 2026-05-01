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

"""Tests for image pipe input extraction."""

import json
import pathlib

import pytest

from cosmos_curator.pipelines.image.read_write.image_writer_stage import get_image_output_id
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask
from cosmos_curator.pipelines.image.utils.image_pipe_input import (
    IMAGE_EXTENSIONS,
    _is_image_file,
    extract_image_tasks,
)


class TestIsImageFile:
    """Tests for _is_image_file."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("a.jpg", True),
            ("a.JPG", True),
            ("a.jpeg", True),
            ("a.png", True),
            ("a.webp", True),
            ("sub/a.jpg", True),
            ("a.txt", False),
            ("a.mp4", False),
            ("a", False),
        ],
    )
    def test_extension_check(self, path: str, expected: bool) -> None:  # noqa: FBT001
        """Only image extensions return True (case-insensitive)."""
        assert _is_image_file(path) is expected


class TestExtractImageTasks:
    """Tests for extract_image_tasks."""

    def test_empty_dir_returns_empty_list(self, tmp_path: pathlib.Path) -> None:
        """Empty directory yields no tasks."""
        tasks = extract_image_tasks(str(tmp_path), "default", verbose=False)
        assert tasks == []

    def test_discovers_images_excludes_non_images(self, tmp_path: pathlib.Path) -> None:
        """Only image extensions are included; others are excluded."""
        (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "b.png").write_bytes(b"\x89PNG")
        (tmp_path / "c.txt").write_text("text")
        (tmp_path / "d.mp4").write_bytes(b"\x00\x00")
        (tmp_path / "e.JPEG").write_bytes(b"\xff\xd8")
        tasks = extract_image_tasks(str(tmp_path), "default", verbose=False)
        assert len(tasks) == 3
        rel_paths = {t.image.relative_path for t in tasks}
        assert rel_paths == {"a.jpg", "b.png", "e.JPEG"}
        for t in tasks:
            assert isinstance(t, ImagePipeTask)
            assert t.session_id
            assert t.image.input_image.exists()

    def test_limit_caps_count(self, tmp_path: pathlib.Path) -> None:
        """Limit > 0 returns at most that many tasks (first after sort)."""
        (tmp_path / "1.jpg").write_bytes(b"1")
        (tmp_path / "2.jpg").write_bytes(b"2")
        (tmp_path / "3.jpg").write_bytes(b"3")
        tasks_all = extract_image_tasks(str(tmp_path), "default", limit=0)
        assert len(tasks_all) == 3
        tasks_2 = extract_image_tasks(str(tmp_path), "default", limit=2)
        assert len(tasks_2) == 2
        # Sorted by relative path
        assert [t.image.relative_path for t in tasks_2] == ["1.jpg", "2.jpg"]

    def test_session_id_is_full_path(self, tmp_path: pathlib.Path) -> None:
        """session_id is the full path to the image (for logging/summary)."""
        (tmp_path / "photo.png").write_bytes(b"\x89PNG")
        tasks = extract_image_tasks(str(tmp_path), "default")
        assert len(tasks) == 1
        assert tasks[0].session_id == str(tmp_path / "photo.png")
        assert tasks[0].image.relative_path == "photo.png"

    def test_skips_already_processed_when_summary_has_images_and_filtered_images(self, tmp_path: pathlib.Path) -> None:
        """summary.json with images+filtered_images (current writer format) skips both passed and filtered."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        (in_dir / "a.jpg").write_bytes(b"\xff\xd8\xff")
        (in_dir / "b.jpg").write_bytes(b"\xff\xd8\xff")
        (in_dir / "c.jpg").write_bytes(b"\xff\xd8\xff")
        out_id_a = get_image_output_id(str(in_dir / "a.jpg"))
        out_id_b = get_image_output_id(str(in_dir / "b.jpg"))
        summary = {"images": [out_id_a], "filtered_images": [out_id_b], "num_input_images": 3}
        (out_dir / "summary.json").write_text(json.dumps(summary))

        tasks = extract_image_tasks(
            str(in_dir),
            "default",
            output_path_and_profile=(str(out_dir), None),
            verbose=False,
        )
        assert len(tasks) == 1
        assert tasks[0].image.relative_path == "c.jpg"

    def test_skips_already_processed_when_summary_has_processed_images(self, tmp_path: pathlib.Path) -> None:
        """Backward compat: summary.json with processed_images is still honoured."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        (in_dir / "a.jpg").write_bytes(b"\xff\xd8\xff")
        (in_dir / "b.jpg").write_bytes(b"\xff\xd8\xff")
        session_id_a = str(in_dir / "a.jpg")
        out_id_a = get_image_output_id(session_id_a)
        summary = {"processed_images": [out_id_a], "num_input_images": 2}
        (out_dir / "summary.json").write_text(json.dumps(summary))

        tasks = extract_image_tasks(
            str(in_dir),
            "default",
            output_path_and_profile=(str(out_dir), None),
            verbose=False,
        )
        assert len(tasks) == 1
        assert tasks[0].image.relative_path == "b.jpg"

    def test_skips_already_processed_when_summary_has_legacy_captioned_images(self, tmp_path: pathlib.Path) -> None:
        """Backward compat: summary.json with captioned_images is still honoured."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        (in_dir / "a.jpg").write_bytes(b"\xff\xd8\xff")
        (in_dir / "b.jpg").write_bytes(b"\xff\xd8\xff")
        session_id_a = str(in_dir / "a.jpg")
        out_id_a = get_image_output_id(session_id_a)
        summary = {"captioned_images": [out_id_a], "num_input_images": 2}
        (out_dir / "summary.json").write_text(json.dumps(summary))

        tasks = extract_image_tasks(
            str(in_dir),
            "default",
            output_path_and_profile=(str(out_dir), None),
            verbose=False,
        )
        assert len(tasks) == 1
        assert tasks[0].image.relative_path == "b.jpg"

    def test_skips_already_processed_when_metas_file_exists(self, tmp_path: pathlib.Path) -> None:
        """Any metas/{id}.json file is sufficient to skip — has_caption is no longer required."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        (out_dir / "metas").mkdir(parents=True)
        (in_dir / "only.jpg").write_bytes(b"\xff\xd8\xff")
        session_id = str(in_dir / "only.jpg")
        out_id = get_image_output_id(session_id)
        meta = {"has_caption": False, "is_filtered": True, "source_path": str(in_dir / "only.jpg")}
        (out_dir / "metas" / f"{out_id}.json").write_text(json.dumps(meta))

        tasks = extract_image_tasks(
            str(in_dir),
            "default",
            output_path_and_profile=(str(out_dir), None),
            verbose=False,
        )
        assert len(tasks) == 0


def test_image_extensions_constant() -> None:
    """IMAGE_EXTENSIONS includes expected suffixes."""
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS
    assert ".webp" in IMAGE_EXTENSIONS
    assert ".mp4" not in IMAGE_EXTENSIONS
