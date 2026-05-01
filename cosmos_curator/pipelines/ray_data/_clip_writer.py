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

"""Clip writer for Ray Data pipelines.

Writes transcoded clip bytes and per-clip JSON metadata to local or remote
storage.
"""

import json
from collections.abc import Callable
from typing import Any

from cosmos_curator.core.utils.storage.storage_utils import StorageWriter
from cosmos_curator.pipelines.video.utils.decoder_utils import extract_video_metadata


def make_write_fn(output_dir: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a ``map`` function that writes a clip and its metadata to storage.

    Each clip produces two artifacts under *output_dir*:

    * ``clips/{clip_uuid}.mp4`` — transcoded clip bytes.
    * ``metas/v0/{clip_uuid}.json`` — per-clip metadata (source video info,
      clip span, and metadata extracted from the transcoded bytes).

    Caption/filter fields are stubbed to empty/zero values — they will be
    populated once downstream stages exist.

    Args:
        output_dir: Base output directory (local path, ``s3://...``, or
            ``az://...``).

    Returns:
        A function suitable for ``ray.data.Dataset.map``.

    """
    writer: StorageWriter | None = None

    def _write(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal writer
        if writer is None:
            writer = StorageWriter(output_dir)

        clip_uuid: str = row["clip_uuid"]
        clip_bytes: bytes = row["clip_bytes"]
        clip_sub_path = f"clips/{clip_uuid}.mp4"
        clip_location = f"{output_dir}/{clip_sub_path}"

        writer.write_bytes_to(clip_sub_path, clip_bytes)

        clip_metadata = extract_video_metadata(clip_bytes)
        metadata: dict[str, Any] = {
            "span_uuid": clip_uuid,
            "source_video": row["video_path"],
            "duration_span": [row["clip_start_s"], row["clip_end_s"]],
            "width_source": row["width_source"],
            "height_source": row["height_source"],
            "framerate_source": row["framerate_source"],
            "clip_location": clip_location,
            "width": clip_metadata.width,
            "height": clip_metadata.height,
            "framerate": clip_metadata.fps,
            "num_frames": clip_metadata.num_frames,
            "video_codec": clip_metadata.video_codec,
            "num_bytes": len(clip_bytes),
            "windows": [],
            "filtered_windows": [],
            "valid": False,
            "has_caption": False,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
        }
        writer.write_str_to(f"metas/v0/{clip_uuid}.json", json.dumps(metadata, indent=4))

        return {
            "video_path": row["video_path"],
            "video_size": row["video_size"],
            "duration_s": row["duration_s"],
            "clip_uuid": clip_uuid,
            "clip_start_s": row["clip_start_s"],
            "clip_end_s": row["clip_end_s"],
            "clip_location": clip_location,
        }

    return _write
