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
"""Sample complete video payloads for storage tests."""

from typing import Any

_FRAME_ROWS: list[dict[str, Any]] = [
    {"dts_ticks": 12, "composition_offset_ticks": None, "is_keyframe": True, "offset": 0, "size": 100},
    {"dts_ticks": 422, "composition_offset_ticks": 6, "is_keyframe": False, "offset": 100, "size": 50},
    {"dts_ticks": 832, "composition_offset_ticks": None, "is_keyframe": False, "offset": 150, "size": 42},
]

_METADATA_ROW: dict[str, Any] = {
    "codec_name": "h264",
    "extradata": b"\x01\x64\x00\x1f",
    "time_base_num": 1,
    "time_base_den": 12288,
    "frame_count": 3,
    "keyframe_count": 1,
    "width": 640,
    "height": 480,
    "avc_profile_idc": 100,
    "avc_constraint_flags": 0,
    "avc_level_idc": 31,
    "nal_length_size": 4,
    "chroma_format_idc": 1,
    "bit_depth_luma_minus8": 0,
    "bit_depth_chroma_minus8": 0,
    "buffer_size_db": None,
    "max_bitrate_bps": None,
    "avg_bitrate_bps": None,
    "movie_creation_time": 0,
    "movie_modification_time": 0,
    "track_creation_time": 0,
    "track_modification_time": 0,
    "track_matrix": [0] * 9,
    "track_display_width": 0,
    "track_display_height": 0,
    "elst_bytes": None,
    "avg_frame_rate_num": 30,
    "avg_frame_rate_den": 1,
    "bit_rate_bps": 1000,
    "duration_ns": 1_000_000_000,
    "encoder_tag": None,
}
