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

"""Video reader for Ray Data pipelines.

Downloads video bytes from local or remote storage and extracts metadata
via ffprobe.
"""

from typing import Any

from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.pipelines.video.utils.decoder_utils import extract_video_metadata


def read_video(row: dict[str, Any]) -> dict[str, Any]:
    """Read a video file and extract metadata.

    Downloads bytes from local/S3/Azure and runs ffprobe for metadata.
    The raw bytes are carried forward as Arrow ``large_binary`` so that
    downstream stages (splitter, transcoder) can access them without a
    second download.

    Args:
        row: Dict with ``"video_path"`` (str).

    Returns:
        Dict with ``video_bytes`` and metadata columns.

    """
    video_path: str = row["video_path"]
    video_bytes = storage_utils.read_bytes(video_path)
    metadata = extract_video_metadata(video_bytes)

    return {
        "video_path": video_path,
        "video_bytes": video_bytes,
        "video_size": len(video_bytes),
        "duration_s": metadata.video_duration,
        "fps": metadata.fps,
        "height": metadata.height,
        "width": metadata.width,
        "video_codec": metadata.video_codec,
        "bit_rate_k": metadata.bit_rate_k,
    }
