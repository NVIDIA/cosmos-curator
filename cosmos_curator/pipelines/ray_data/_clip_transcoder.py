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

"""Clip transcoder for Ray Data pipelines.

Transcodes all clip segments for a single video and fans out to one row per
clip.  Designed for use with ``flat_map`` — each call receives one video row
(with clip spans as lists) and returns N clip rows with transcoded bytes.
"""

import logging
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BITRATE = "4M"


def make_transcode_fn(
    encoder: str = "libopenh264",
    encoder_threads: int = 1,
    ffmpeg_batch_size: int = 16,
    *,
    use_input_bit_rate: bool = False,
) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    """Create a ``flat_map`` function that transcodes clips for one video.

    The function receives a single video row containing ``video_bytes`` and
    clip span lists (``clip_uuids``, ``clip_starts``, ``clip_ends``), writes
    the video to a temp file once, runs batched FFmpeg commands, and returns
    one row per successfully transcoded clip.

    Args:
        encoder: FFmpeg video encoder (``"libopenh264"``).
        encoder_threads: Threads per FFmpeg encoding sub-command.
        ffmpeg_batch_size: Max clips per single FFmpeg invocation.
        use_input_bit_rate: If ``True``, use the source video's bitrate
            instead of the default 4 Mbps.

    Returns:
        A function suitable for ``ray.data.Dataset.flat_map``.

    """

    def _transcode(row: dict[str, Any]) -> list[dict[str, Any]]:
        video_path: str = row["video_path"]
        video_bytes: bytes = row["video_bytes"]
        video_size: int = row["video_size"]
        duration_s: float = row["duration_s"]
        clip_uuids: list[str] = row["clip_uuids"]
        clip_starts: list[float] = row["clip_starts"]
        clip_ends: list[float] = row["clip_ends"]
        width_source: int = row["width"]
        height_source: int = row["height"]
        framerate_source: float = row["fps"]

        bitrate = f"{row['bit_rate_k']}K" if use_input_bit_rate and "bit_rate_k" in row else _DEFAULT_BITRATE

        results: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="ray_data_transcode_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write the source video to disk once.
            video_file = tmp_path / "input.mp4"
            video_file.write_bytes(video_bytes)

            # Sub-batch clips for FFmpeg invocations.
            for sub_start in range(0, len(clip_uuids), ffmpeg_batch_size):
                sub_end = min(sub_start + ffmpeg_batch_size, len(clip_uuids))
                _run_ffmpeg(
                    tmp_path,
                    video_file.name,
                    clip_uuids[sub_start:sub_end],
                    clip_starts[sub_start:sub_end],
                    clip_ends[sub_start:sub_end],
                    bitrate,
                    encoder,
                    encoder_threads,
                )

                # Read back transcoded clip bytes.
                for i in range(sub_start, sub_end):
                    clip_file = tmp_path / f"{clip_uuids[i]}.mp4"
                    if clip_file.exists():
                        results.append(
                            {
                                "video_path": video_path,
                                "video_size": video_size,
                                "duration_s": duration_s,
                                "clip_uuid": clip_uuids[i],
                                "clip_start_s": clip_starts[i],
                                "clip_end_s": clip_ends[i],
                                "clip_bytes": clip_file.read_bytes(),
                                "width_source": width_source,
                                "height_source": height_source,
                                "framerate_source": framerate_source,
                            }
                        )
                        clip_file.unlink()
                    else:
                        logger.warning("Transcoded clip file missing for %s", clip_uuids[i])

        return results

    return _transcode


def _run_ffmpeg(  # noqa: PLR0913
    working_dir: Path,
    video_filename: str,
    clip_uuids: list[str],
    clip_starts: list[float],
    clip_ends: list[float],
    bitrate: str,
    encoder: str,
    encoder_threads: int,
) -> None:
    """Run a single batched FFmpeg command to transcode clips.

    Args:
        working_dir: Temporary directory containing the source video.
        video_filename: Filename of the source video in *working_dir*.
        clip_uuids: UUIDs for clips in this sub-batch.
        clip_starts: Start times for clips in this sub-batch.
        clip_ends: End times for clips in this sub-batch.
        bitrate: Target bitrate string (e.g. ``"4M"`` or ``"4000K"``).
        encoder: FFmpeg video encoder name.
        encoder_threads: Threads per encoding sub-command.

    """
    command: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    for i, (clip_uuid, start_s, end_s) in enumerate(zip(clip_uuids, clip_starts, clip_ends, strict=True)):
        command.extend(["-threads", str(encoder_threads)])
        command.extend(
            [
                "-ss",
                str(start_s),
                "-to",
                str(end_s),
                "-i",
                video_filename,
                "-map",
                f"{i}:v:0",
                "-c:v",
                encoder,
                "-b:v",
                bitrate,
            ]
        )
        command.extend(["-threads", str(encoder_threads)])
        command.extend(
            [
                "-map",
                f"{i}:a:0?",
                "-c:a",
                "copy",
                f"{clip_uuid}.mp4",
            ]
        )

    try:
        subprocess.check_output(command, cwd=working_dir, stderr=subprocess.STDOUT)  # noqa: S603
    except subprocess.CalledProcessError as e:
        output_text = e.output.decode("utf-8") if e.output else str(e)
        logger.exception("FFmpeg failed (rc=%d): %s", e.returncode, output_text)
