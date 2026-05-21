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
"""Remux helpers for video pipelines."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.utils.data_model import Video

REMUX_FORMATS = {"mpegts"}


def remux_to_mp4(encoded_data: bytes | npt.NDArray[np.uint8], threads: int = 1) -> npt.NDArray[np.uint8]:
    """Remux a video to a MP4 container using ffmpeg.

    Notes:
    A moov atom is placed at the beginning of the file. This atom contains
    information about the video, such as the duration, number of frames,
    time base, etc.

    By default, ffmpeg will place the moov atom at the end of the file, and
    the entire stream must be read to the end to start decoding.

    By placing the moov atom at the beginning, decode can start immediately.

    To achieve this, FFmpeg transcodes the entire stream, and then writes the
    moov atom at the end.

    And then the moov atom is moved to the beginning of the file, which
    requires that the output file is seekable.

    Thus, we incur overhead of file io when writing to the temp file instead
    of being able to operate purely in memory.

    Args:
        encoded_data: The bytes of the input video (e.g., MPEG-TS).
        threads: The number of threads to use for ffmpeg.

    Returns:
        The bytes of the remuxed MP4 video.

    """
    # ffmpeg needs the output to be seekable, so write ffmpeg output to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as output_file:
        cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            f"{threads}",
            "-fflags",
            "+genpts",
            "-i",
            "-",  # Read from stdin
            "-c",
            "copy",  # Copy streams without re-encoding
            "-movflags",
            "+faststart",  # Place the moov atom at the beginning of the file
            "-f",
            "mp4",  # Force MP4 format (since stdout is ambiguous)
            f"{output_file.name}",  # Write to disk - ffmpeg needs output to be seekable
        ]

        logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")
        stdin_data = encoded_data if isinstance(encoded_data, bytes) else encoded_data.tobytes()
        proc = subprocess.run(  # noqa: S603
            cmd, input=stdin_data, capture_output=True, check=False
        )

        if proc.returncode != 0:
            msg = f"ffmpeg failed with return code {proc.returncode}:\n{proc.stderr.decode('utf-8')}"
            raise RuntimeError(msg)

        stderr_output = proc.stderr.decode("utf-8", errors="replace")
        logger.debug(f"ffmpeg stderr:\n{stderr_output}")
        return bytes_to_numpy(Path(output_file.name).read_bytes())


def remux_if_needed(video: Video, threads: int) -> bool:
    """Remux the video if it is not in the correct format.

    Args:
        video: The video to remux, modified in place.
        threads: The number of threads to use for ffmpeg.

    Returns:
        True if a remux was applied and succeeded.
        False if no remux was needed (format not in REMUX_FORMATS) or metadata is absent.
        Raises on precondition failure (encoded_data is None) or remux failure — caller
        handles via try/except.

    """
    data = video.encoded_data.resolve()
    if data is None:
        msg = "Video source bytes are not set"
        raise ValueError(msg)

    if not video.metadata:
        logger.warning(f"Video {video.input_video} has no metadata, skipping remux")
        # TODO(LazyData): re-enable .release() when .store() is active.
        # Without .store(), .release() clears the only copy -> ValueError downstream.
        # video.encoded_data.release()  # noqa: ERA001
        return False

    format_name = video.metadata.format_name.lower() if video.metadata.format_name else "unknown"

    if any(remux_format in format_name for remux_format in REMUX_FORMATS):
        logger.info(f"Video {video.input_video} is in `{format_name}` format, remuxing to mp4")
        video.encoded_data = remux_to_mp4(data, threads=threads)  # type: ignore[assignment]
        video.timestamps = (
            None  # Invalidate after byte replacement; keep state correct even if populate_metadata() fails.
        )
        video.populate_metadata()
        # TODO(LazyData): re-enable when batch-mode ObjectRef ownership is
        # resolved.  In batch mode, pool.stop() kills actor -> OwnerDiedError.
        # video.encoded_data.store()  # noqa: ERA001
        return True
    # TODO(LazyData): re-enable .release() when .store() is active.
    # Without .store(), .release() clears the only copy -> ValueError downstream.
    # video.encoded_data.release()  # noqa: ERA001
    return False
