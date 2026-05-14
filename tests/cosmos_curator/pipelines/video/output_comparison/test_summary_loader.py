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
"""Tests for split output summary loading."""

from pathlib import Path

import pytest

from cosmos_curator.pipelines.video.output_comparison.summary_loader import load_summary

from .conftest import summary, write_summary


def test_load_summary_returns_typed_summary(tmp_path: Path) -> None:
    """Loading a valid summary returns the typed summary object."""
    output_root = tmp_path / "output"
    write_summary(output_root, summary())

    loaded_summary = load_summary(output_root, profile_name="default")

    assert loaded_summary.num_input_videos == 1
    assert tuple(loaded_summary.videos) == ("video.mp4",)


def test_load_summary_raises_for_missing_summary(tmp_path: Path) -> None:
    """Missing summary.json bubbles out of the loader."""
    output_root = tmp_path / "output"
    output_root.mkdir()

    with pytest.raises(FileNotFoundError):
        load_summary(output_root, profile_name="default")


def test_load_summary_raises_for_non_object_summary(tmp_path: Path) -> None:
    """A non-object summary.json bubbles out as a validation error."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "summary.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=r"summary\.json must contain a JSON object with string keys"):
        load_summary(output_root, profile_name="default")
