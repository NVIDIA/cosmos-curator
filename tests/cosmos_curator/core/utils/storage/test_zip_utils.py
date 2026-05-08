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
"""Tests for zip archive helpers."""

import stat
import zipfile
from pathlib import Path

import pytest

from cosmos_curator.core.utils.storage.zip_utils import safe_extract_zip


def test_safe_extract_zip_extracts_regular_members(tmp_path: Path) -> None:
    """Extract regular archive members."""
    archive = tmp_path / "archive.zip"
    dest = tmp_path / "dest"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/file.txt", "payload")

    with zipfile.ZipFile(archive) as zf:
        safe_extract_zip(zf, dest)

    assert (dest / "nested" / "file.txt").read_text(encoding="utf-8") == "payload"


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    """Reject archive members that would write outside the destination."""
    archive = tmp_path / "archive.zip"
    dest = tmp_path / "dest"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "payload")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError, match="outside destination"):
        safe_extract_zip(zf, dest)

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_zip_rejects_symlink_members(tmp_path: Path) -> None:
    """Reject symlink archive entries."""
    archive = tmp_path / "archive.zip"
    dest = tmp_path / "dest"
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, "target")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError, match="symlink"):
        safe_extract_zip(zf, dest)


def test_safe_extract_zip_rejects_existing_symlinked_parent(tmp_path: Path) -> None:
    """Reject members that would escape through an existing symlinked directory."""
    archive = tmp_path / "archive.zip"
    dest = tmp_path / "dest"
    outside = tmp_path / "outside"
    dest.mkdir()
    outside.mkdir()
    try:
        (dest / "linked").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are not available: {exc}")

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("linked/escape.txt", "payload")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError, match="outside destination"):
        safe_extract_zip(zf, dest)

    assert not (outside / "escape.txt").exists()
