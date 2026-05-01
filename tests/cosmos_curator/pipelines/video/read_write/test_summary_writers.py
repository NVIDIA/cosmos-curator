# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for write_split_summary num_remuxed_videos metric."""

import pathlib
from unittest.mock import patch

from cosmos_curator.pipelines.video.read_write.summary_writers import write_split_summary
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask, Video


def _make_video(*, was_remuxed: bool, clip_chunk_index: int) -> Video:
    v = Video(input_video=pathlib.Path("test.ts"))
    v.was_remuxed = was_remuxed
    v.clip_chunk_index = clip_chunk_index
    return v


def test_num_remuxed_videos_no_double_count() -> None:
    """clip_chunk_index == 0 guard prevents double-counting chunked videos.

    Two Video objects represent the same source video split into two chunks.
    Both have was_remuxed=True, but only the chunk-0 object should be counted,
    so num_remuxed_videos must be 1, not 2.
    """
    chunk0 = _make_video(was_remuxed=True, clip_chunk_index=0)
    chunk1 = _make_video(was_remuxed=True, clip_chunk_index=1)

    task0 = SplitPipeTask(session_id="s", video=chunk0)
    task1 = SplitPipeTask(session_id="s", video=chunk1)

    with patch("cosmos_curator.pipelines.video.read_write.summary_writers._write_split_result_summary") as mock_write:
        write_split_summary(
            input_path="/in",
            input_videos_relative=["test.ts"],
            num_input_videos_selected=1,
            output_path="/out",
            output_s3_profile_name="default",
            output_tasks=[task0, task1],
            embedding_algorithm="internvideo2",
            limit=0,
        )

    mock_write.assert_called_once()
    _, kwargs = mock_write.call_args
    assert kwargs["num_remuxed_videos"] == 1, (
        f"Expected 1 remuxed video (chunk-0 only), got {kwargs['num_remuxed_videos']}"
    )
