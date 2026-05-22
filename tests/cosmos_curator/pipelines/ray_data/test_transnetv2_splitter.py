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

"""Tests for the Ray Data TransNetV2 splitter helpers."""

import uuid
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.pipelines.ray_data import _transnetv2_splitter as _splitter


def test_split_transnetv2_frames_outputs_clip_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """TransNetV2 spans should match the downstream Ray Data clip-column contract."""

    def fake_get_predictions(
        _model: object,
        frames: npt.NDArray[np.uint8],
        _threshold: float,
    ) -> npt.NDArray[np.uint8]:
        predictions = np.zeros((len(frames), 1), dtype=np.uint8)
        predictions[50] = 1
        return predictions

    monkeypatch.setattr(_splitter, "_get_predictions", fake_get_predictions)
    frames = np.zeros((100, 27, 48, 3), dtype=np.uint8)
    video_path = "s3://bucket/raw/video.mp4"

    clip_uuids, clip_starts, clip_ends = _splitter.split_transnetv2_frames(
        frames=frames,
        video_path=video_path,
        fps=25.0,
        model=lambda _batch: None,  # type: ignore[arg-type, return-value]
        threshold=0.4,
        min_length_s=None,
        min_length_frames=None,
        max_length_s=None,
        max_length_mode="stride",
        crop_s=0.0,
        entire_scene_as_clip=True,
        limit_clips=1,
    )

    assert clip_uuids == [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_path}_0_50"))]
    assert clip_starts == [0.0]
    assert clip_ends == [2.0]


def test_split_transnetv2_frames_uses_full_scene_when_no_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-transition behavior should preserve Xenna's full-scene default."""

    def fake_get_predictions(
        _model: object,
        frames: npt.NDArray[np.uint8],
        _threshold: float,
    ) -> npt.NDArray[np.uint8]:
        return np.zeros((len(frames), 1), dtype=np.uint8)

    monkeypatch.setattr(_splitter, "_get_predictions", fake_get_predictions)
    frames = np.zeros((80, 27, 48, 3), dtype=np.uint8)

    _, clip_starts, clip_ends = _splitter.split_transnetv2_frames(
        frames=frames,
        video_path="/videos/a.mp4",
        fps=20.0,
        model=lambda _batch: None,  # type: ignore[arg-type, return-value]
        threshold=1.0,
        min_length_s=None,
        min_length_frames=None,
        max_length_s=None,
        max_length_mode="stride",
        crop_s=0.0,
        entire_scene_as_clip=True,
        limit_clips=0,
    )

    assert clip_starts == [0.0]
    assert clip_ends == [4.0]


def test_split_transnetv2_frames_rejects_unexpected_frame_shape() -> None:
    """TransNetV2 requires the same 27x48 RGB frame shape as the Xenna stage."""
    frames = np.zeros((10, 28, 48, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected frames of shape 27x48x3"):
        _splitter.split_transnetv2_frames(
            frames=frames,
            video_path="/videos/a.mp4",
            fps=20.0,
            model=lambda _batch: None,  # type: ignore[arg-type, return-value]
            threshold=0.4,
            min_length_s=None,
            min_length_frames=None,
            max_length_s=None,
            max_length_mode="stride",
            crop_s=0.0,
            entire_scene_as_clip=True,
            limit_clips=0,
        )


def test_transnetv2_splitter_aligns_ffmpeg_threads_with_reserved_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FFmpeg should not request more threads than the Ray actor reserves."""

    class FakeTransNetV2:
        def setup(self) -> None:
            """Skip model setup."""

    monkeypatch.setattr(_splitter.transnetv2, "TransNetV2", FakeTransNetV2)
    monkeypatch.setattr(_splitter, "gpu_stage_cleanup", lambda _stage_name: None)

    splitter = _splitter.TransNetV2Splitter(num_decode_cpus_per_worker=2)

    assert splitter._num_cpu_threads == 2


def test_transnetv2_splitter_rejects_max_length_below_min_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ray Data TransNetV2 should reject impossible length bounds like the Xenna stage."""

    class FakeTransNetV2:
        def setup(self) -> None:
            """Fail if validation does not run before model setup."""
            pytest.fail("model setup should not run for invalid length bounds")

    monkeypatch.setattr(_splitter.transnetv2, "TransNetV2", FakeTransNetV2)

    with pytest.raises(ValueError, match="Max length is smaller than min length"):
        _splitter.TransNetV2Splitter(min_length_s=10.0, max_length_s=5.0)


def test_transnetv2_splitter_emits_empty_columns_when_fps_is_invalid() -> None:
    """Invalid source FPS should skip the video before frame-to-second math."""
    splitter = _splitter.TransNetV2Splitter.__new__(_splitter.TransNetV2Splitter)
    splitter._decode_frames = lambda _row: pytest.fail("_decode_frames should not be called")
    splitter._model = object()

    row: dict[str, Any] = {
        "video_path": "/videos/a.mp4",
        "video_bytes": b"video",
        "fps": 0.0,
    }

    output = splitter(row)

    assert output["clip_uuids"] == []
    assert output["clip_starts"] == []
    assert output["clip_ends"] == []


def test_transnetv2_splitter_emits_empty_columns_when_decode_fails() -> None:
    """Decode failures should skip the video without breaking downstream stages."""
    splitter = _splitter.TransNetV2Splitter.__new__(_splitter.TransNetV2Splitter)
    splitter._decode_frames = lambda _row: None
    splitter._model = object()

    row: dict[str, Any] = {
        "video_path": "/videos/a.mp4",
        "video_bytes": b"bad-video",
        "fps": 30.0,
    }

    output = splitter(row)

    assert output["clip_uuids"] == []
    assert output["clip_starts"] == []
    assert output["clip_ends"] == []
