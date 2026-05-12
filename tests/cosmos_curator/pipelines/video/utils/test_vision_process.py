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
"""Tests for video vision processing helpers."""

from typing import Any

import numpy as np
import pytest

from cosmos_curator.pipelines.video.utils import vision_process
from cosmos_curator.pipelines.video.utils.windowing_utils import WindowFrameInfo


def _patch_video_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(vision_process, "get_avg_frame_rate", lambda _video_path: 10.0)

    def fake_smart_nframes(fps: float, total_frames: int, video_fps: float) -> int:
        del fps, video_fps
        captured.setdefault("total_frames", []).append(total_frames)
        return total_frames

    def fake_decode_video_cpu_frame_ids(_video_path: str, frame_ids: list[int]) -> np.ndarray:
        captured["frame_ids"] = frame_ids
        return np.zeros((len(frame_ids), 2, 3, 3), dtype=np.uint8)

    monkeypatch.setattr(vision_process, "smart_nframes", fake_smart_nframes)
    monkeypatch.setattr(vision_process, "decode_video_cpu_frame_ids", fake_decode_video_cpu_frame_ids)
    return captured


def test_read_video_cpu_treats_window_end_as_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inclusive window end is sampled and counted as a native frame."""
    captured = _patch_video_reader(monkeypatch)

    video, frame_counts = vision_process.read_video_cpu(
        "video.mp4",
        fps=10.0,
        num_frames_to_use=0,
        window_range=[WindowFrameInfo(start=10, end=14)],
    )

    assert captured["total_frames"] == [5]
    assert captured["frame_ids"] == [10, 11, 12, 13, 14]
    assert frame_counts == [5]
    assert tuple(video.shape) == (5, 3, 2, 3)


def test_read_video_cpu_num_frames_to_use_preserves_inclusive_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frame caps sample from start through start + cap - 1."""
    captured = _patch_video_reader(monkeypatch)

    vision_process.read_video_cpu(
        "video.mp4",
        fps=10.0,
        num_frames_to_use=4,
        window_range=[WindowFrameInfo(start=10, end=19)],
    )

    assert captured["total_frames"] == [4]
    assert captured["frame_ids"] == [10, 11, 12, 13]
