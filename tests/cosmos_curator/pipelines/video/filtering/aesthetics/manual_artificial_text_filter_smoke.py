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
"""Smoke test for artificial text filter stage (manual run only; not in default CI).

Run manually from inside the Docker image with the paddle-ocr env. You must pass
-m env so these tests are selected (default pytest addopts deselect env-marked tests):

pixi run --as-is -e paddle-ocr pytest -m env  \
tests/cosmos_curator/pipelines/video/filtering/aesthetics/manual_artificial_text_filter_smoke.py

Requires test_clip_10s.mp4 from tests/.../video/data/ (or sample_clip_data fixture).
"""

import pytest

from cosmos_curator.core.interfaces.pipeline_interface import run_pipeline
from cosmos_curator.core.interfaces.runner_interface import RunnerInterface
from cosmos_curator.pipelines.video.filtering.aesthetics.artificial_text_filter_stage import (
    ArtificialTextFilterStage,
)
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask

# Golden values for test_clip_10s.mp4
EXPECTED_HAS_ARTIFICIAL_TEXT: bool = True
EXPECTED_ARTIFICIAL_TEXT_SEGMENT_COUNT: int = 2


@pytest.mark.env("paddle-ocr")
def test_artificial_text_filter_setup() -> None:
    """Stage sets up and tears down in paddle-ocr env."""
    stage = ArtificialTextFilterStage(
        num_gpus_per_worker=0.25,
        use_corner_detection=False,
        frame_interval=3,
        verbose=False,
        log_stats=True,
    )
    stage.stage_setup()
    assert stage.model is not None
    stage.destroy()


@pytest.mark.env("paddle-ocr")
def test_artificial_text_filter_process(
    sample_filtering_task: SplitPipeTask,
    sequential_runner: RunnerInterface,
) -> None:
    """Run full pipeline on test_clip_10s.mp4; assert structure and golden outcome."""
    stage = ArtificialTextFilterStage(
        num_gpus_per_worker=0.25,
        use_corner_detection=True,
        frame_interval=3,
        verbose=True,
        log_stats=True,
    )
    result_tasks = run_pipeline(
        [sample_filtering_task],
        [stage],
        runner=sequential_runner,
    )

    assert len(result_tasks) == 1
    video = result_tasks[0].video
    total = len(video.clips) + len(video.filtered_clips)
    assert total == 1

    # Single clip: either kept or filtered
    all_clips = video.clips + video.filtered_clips
    assert len(all_clips) == 1
    clip = all_clips[0]

    assert clip.has_artificial_text is not None
    assert clip.has_artificial_text == EXPECTED_HAS_ARTIFICIAL_TEXT, (
        "Golden value mismatch: run once in paddle-ocr and update "
        "EXPECTED_HAS_ARTIFICIAL_TEXT / EXPECTED_ARTIFICIAL_TEXT_SEGMENT_COUNT"
    )
    if clip.has_artificial_text:
        assert isinstance(clip.artificial_text_segments, list)
        assert len(clip.artificial_text_segments) == EXPECTED_ARTIFICIAL_TEXT_SEGMENT_COUNT
    else:
        assert clip.artificial_text_segments is None
        assert EXPECTED_ARTIFICIAL_TEXT_SEGMENT_COUNT == 0

    assert "ArtificialTextFilterStage" in result_tasks[0].stage_perf
