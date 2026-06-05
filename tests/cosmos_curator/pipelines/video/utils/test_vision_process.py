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
import torch

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


def _patch_fetch_video_resize(monkeypatch: pytest.MonkeyPatch, nframes: int) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_read_video_cpu(
        _video_path: str,
        _fps: float,
        _num_frames_to_use: int,
        _window_range: list[WindowFrameInfo],
    ) -> tuple[torch.Tensor, list[int]]:
        return torch.zeros((nframes, 3, 120, 200), dtype=torch.uint8), [nframes]

    def fake_smart_resize(
        height: int,
        width: int,
        *,
        factor: int,
        min_pixels: int,
        max_pixels: int,
    ) -> tuple[int, int]:
        captured["smart_resize"] = {
            "height": height,
            "width": width,
            "factor": factor,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
        return height, width

    monkeypatch.setattr(vision_process, "read_video_cpu", fake_read_video_cpu)
    monkeypatch.setattr(vision_process, "smart_resize", fake_smart_resize)
    monkeypatch.setattr(vision_process.transforms.functional, "resize", lambda video, *_args, **_kwargs: video)
    return captured


def test_fetch_video_unset_uses_dynamic_max_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default sync prep keeps the existing clip-coupled resize formula."""
    nframes = 100
    captured = _patch_fetch_video_resize(monkeypatch, nframes=nframes)

    vision_process.fetch_video("video.mp4")

    expected = max(
        min(
            vision_process.VIDEO_MAX_PIXELS,
            int(vision_process.VIDEO_TOTAL_PIXELS / nframes * vision_process.FRAME_FACTOR),
        ),
        int(vision_process.VIDEO_MIN_PIXELS * 1.05),
    )
    assert captured["smart_resize"]["max_pixels"] == expected


def test_fetch_video_override_uses_exact_max_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A low-but-valid override is passed through without the old 1.05 floor."""
    captured = _patch_fetch_video_resize(monkeypatch, nframes=100)

    vision_process.fetch_video("video.mp4", max_pixels_per_frame=100500)

    assert captured["smart_resize"]["max_pixels"] == 100500
    assert captured["smart_resize"]["min_pixels"] == vision_process.VIDEO_MIN_PIXELS


def test_fetch_video_rejects_uint8_with_curator_preprocess() -> None:
    """Normalized curator-side tensors must stay in a floating dtype."""
    with pytest.raises(ValueError, match=r"preprocess_dtype='uint8' is only valid when do_preprocess=False"):
        vision_process.fetch_video("video.mp4", do_preprocess=True, preprocess_dtype="uint8")


def test_fetch_video_allows_uint8_without_curator_preprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model-side preprocessing paths may still request resized uint8 frames."""
    _patch_fetch_video_resize(monkeypatch, nframes=4)

    video, frame_counts = vision_process.fetch_video("video.mp4", do_preprocess=False, preprocess_dtype="uint8")

    assert video.dtype == torch.uint8
    assert frame_counts == [4]


def test_fetch_video_rejects_unsupported_preprocess_dtype() -> None:
    """Unsupported dtype strings should fail before decode instead of falling back."""
    with pytest.raises(ValueError, match=r"Unsupported preprocess_dtype='float8'"):
        vision_process.fetch_video("video.mp4", preprocess_dtype="float8")
