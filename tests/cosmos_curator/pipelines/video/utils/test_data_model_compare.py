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
"""Tests for video task comparators."""

from pathlib import Path

from cosmos_curator.core.utils.infra.performance_utils import StagePerfStats
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask, Video
from cosmos_curator.pipelines.video.utils.data_model_compare import SplitPipeTaskComparator


def _make_split_task(tmp_path: Path) -> SplitPipeTask:
    """Create a minimal SplitPipeTask fixture."""
    return SplitPipeTask(
        session_id="session-a",
        videos=[Video(input_video=tmp_path / "video.mp4")],
    )


def test_split_pipe_task_comparator_ignores_stage_perf(tmp_path: Path) -> None:
    """Perf metadata should not affect semantic compare results."""
    golden = _make_split_task(tmp_path)
    candidate = _make_split_task(tmp_path)
    candidate.stage_perf["stage-a"] = StagePerfStats()
    assert SplitPipeTaskComparator().compare(golden, candidate, atol=0.0) == []


def test_split_pipe_task_comparator_reports_semantic_difference(tmp_path: Path) -> None:
    """Semantic task changes should still be reported."""
    golden = _make_split_task(tmp_path)
    candidate = _make_split_task(tmp_path)
    candidate.errors["stage-a"] = "boom"
    failures = SplitPipeTaskComparator().compare(golden, candidate, atol=0.0)
    assert len(failures) == 1
    assert failures[0].field == "errors"
