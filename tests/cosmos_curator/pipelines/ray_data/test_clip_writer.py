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

"""Tests for the Ray Data clip writer."""

import json
from pathlib import Path

import pytest

from cosmos_curator.pipelines.ray_data._clip_writer import make_write_fn
from cosmos_curator.pipelines.video.utils.decoder_utils import extract_video_metadata

_FIXTURE_CLIP = Path(__file__).parents[2] / "pipelines" / "video" / "data" / "test_clip_10s.mp4"


@pytest.mark.env("cosmos-curator")
def test_write_emits_mp4_and_json(tmp_path: Path) -> None:
    """Writer emits both ``clips/{uuid}.mp4`` and ``metas/v0/{uuid}.json``."""
    clip_bytes = _FIXTURE_CLIP.read_bytes()
    clip_uuid = "55f3cf21-ce64-587c-b73d-30834b728ff5"
    source_video = "s3://bucket/raw/sample.mp4"
    row = {
        "video_path": source_video,
        "video_size": 123456,
        "duration_s": 600.0,
        "clip_uuid": clip_uuid,
        "clip_start_s": 350.0,
        "clip_end_s": 360.0,
        "clip_bytes": clip_bytes,
        "width_source": 1280,
        "height_source": 720,
        "framerate_source": 29.97,
    }

    result = make_write_fn(str(tmp_path))(row)

    expected_clip_location = f"{tmp_path}/clips/{clip_uuid}.mp4"
    assert result == {
        "video_path": source_video,
        "video_size": 123456,
        "duration_s": 600.0,
        "clip_uuid": clip_uuid,
        "clip_start_s": 350.0,
        "clip_end_s": 360.0,
        "clip_location": expected_clip_location,
    }

    mp4_path = tmp_path / "clips" / f"{clip_uuid}.mp4"
    assert mp4_path.read_bytes() == clip_bytes

    json_path = tmp_path / "metas" / "v0" / f"{clip_uuid}.json"
    metadata = json.loads(json_path.read_text())

    clip_meta = extract_video_metadata(clip_bytes)
    assert metadata == {
        "span_uuid": clip_uuid,
        "source_video": source_video,
        "duration_span": [350.0, 360.0],
        "width_source": 1280,
        "height_source": 720,
        "framerate_source": 29.97,
        "clip_location": expected_clip_location,
        "width": clip_meta.width,
        "height": clip_meta.height,
        "framerate": clip_meta.fps,
        "num_frames": clip_meta.num_frames,
        "video_codec": clip_meta.video_codec,
        "num_bytes": len(clip_bytes),
        "windows": [],
        "filtered_windows": [],
        "valid": False,
        "has_caption": False,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
    }
