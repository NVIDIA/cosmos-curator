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
"""Tests for the artificial text (overlay/post-production) filter stage.

Unit tests mock PaddleOCR and run in the default env. Integration/smoke tests
are in manual_artificial_text_filter_smoke.py (run manually with paddle-ocr env).
"""

import pathlib
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cosmos_curator.pipelines.video.filtering.aesthetics.artificial_text_filter_stage import (
    ArtificialTextFilterStage,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video


def _make_task(
    *,
    encoded_data: bytes | None = b"dummy",
) -> SplitPipeTask:
    """Build a minimal SplitPipeTask with one clip for stage tests."""
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video="test.mp4",
        span=(0.0, 5.0),
        encoded_data=encoded_data,
    )
    video = Video(
        input_video=pathlib.Path("test.mp4"),
        clips=[clip],
        filtered_clips=[],
    )
    return SplitPipeTask(session_id="test-session", video=video, stage_perf={})


@pytest.fixture
def artificial_text_stage() -> ArtificialTextFilterStage:
    """Stage instance with corner detection disabled for faster unit tests."""
    return ArtificialTextFilterStage(
        num_gpus_per_worker=0.25,
        use_corner_detection=False,
        frame_interval=3,
        verbose=False,
        log_stats=True,
    )


def test_process_data_invalid_dimensions(
    artificial_text_stage: ArtificialTextFilterStage,
) -> None:
    """Clips with invalid dimensions get has_artificial_text=False and error set."""
    task = _make_task()
    fake_meta = SimpleNamespace(
        width=0,
        height=0,
        fps=30.0,
        num_frames=0,
        video_codec="avc1",
    )

    with (
        patch(
            "cosmos_curator.pipelines.video.utils.data_model.extract_video_metadata",
            return_value=fake_meta,
        ),
        patch.object(
            artificial_text_stage._model,
            "generate_single",
            return_value=[[]],
        ),
    ):
        result = artificial_text_stage.process_data([task])

    assert result is not None
    video = result[0].video
    assert video.clips[0].has_artificial_text is False
    assert video.clips[0].errors.get("artificial_text") == "invalid dimensions"


def test_process_data_no_artificial_text(
    artificial_text_stage: ArtificialTextFilterStage,
) -> None:
    """When OCR returns no segments, clip is kept in video.clips."""
    task = _make_task()
    fake_meta = SimpleNamespace(
        width=640,
        height=480,
        fps=30.0,
        num_frames=100,
        video_codec="avc1",
    )
    # Empty boxes per frame -> detector returns no segments
    empty_ocr_results = [[]] * 4

    with (
        patch(
            "cosmos_curator.pipelines.video.utils.data_model.extract_video_metadata",
            return_value=fake_meta,
        ),
        patch.object(
            artificial_text_stage._model,
            "generate_single",
            return_value=empty_ocr_results,
        ),
    ):
        result = artificial_text_stage.process_data([task])

    assert result is not None
    video = result[0].video
    assert len(video.clips) == 1
    assert video.clips[0].has_artificial_text is False
    assert video.clips[0].artificial_text_segments is None
    assert len(video.filtered_clips) == 0
    assert video.clip_stats.num_filtered_by_artificial_text == 0


def test_process_data_has_artificial_text(
    artificial_text_stage: ArtificialTextFilterStage,
) -> None:
    """When detector returns segments, clip is moved to filtered_clips."""
    task = _make_task()
    fake_meta = SimpleNamespace(
        width=640,
        height=480,
        fps=30.0,
        num_frames=100,
        video_codec="avc1",
    )
    # One box on first frame only; we mock the detector to return one segment
    ocr_results = [[[[0, 0], [100, 0], [100, 50], [0, 50]]]] + [[]] * 3
    mock_segment = {"start_frame": 0, "end_frame": 0, "classification_reason": "stable_location_text"}

    with (
        patch(
            "cosmos_curator.pipelines.video.utils.data_model.extract_video_metadata",
            return_value=fake_meta,
        ),
        patch.object(
            artificial_text_stage._model,
            "generate_single",
            return_value=ocr_results,
        ),
        patch(
            "cosmos_curator.pipelines.video.filtering.aesthetics.artificial_text_filter_stage.ArtificialTextDetector",
        ) as mock_detector_cls,
    ):
        mock_detector = mock_detector_cls.return_value
        mock_detector.detect.return_value = [mock_segment]
        result = artificial_text_stage.process_data([task])

    assert result is not None
    video = result[0].video
    assert len(video.clips) == 0
    assert len(video.filtered_clips) == 1
    filtered = video.filtered_clips[0]
    assert filtered.has_artificial_text is True
    assert filtered.artificial_text_segments == [mock_segment]
    assert video.clip_stats.num_filtered_by_artificial_text == 1


def test_process_data_metadata_exception(
    artificial_text_stage: ArtificialTextFilterStage,
) -> None:
    """When extract_metadata raises, clip gets has_artificial_text=False and error set."""
    task = _make_task()

    with (
        patch(
            "cosmos_curator.pipelines.video.utils.data_model.extract_video_metadata",
            side_effect=RuntimeError("decode failed"),
        ),
        patch.object(
            artificial_text_stage._model,
            "generate_single",
            return_value=[[]],
        ),
    ):
        result = artificial_text_stage.process_data([task])

    assert result is not None
    clip = result[0].video.clips[0]
    assert clip.has_artificial_text is False
    assert "artificial_text" in clip.errors
    assert "decode failed" in clip.errors["artificial_text"]
