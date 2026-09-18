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
"""Types for the sensor library."""

import enum
import io
from pathlib import Path

# DataSource is the union of input shapes accepted by sensors and IO helpers.
#
# - Path: a local filesystem path; opened in "rb" mode by the library.
# - bytes: an owned byte buffer; wrapped in ``io.BytesIO`` by the library.
# - io.BufferedIOBase: a caller-owned binary stream. Must be readable and
#   seekable. The sensor library treats it as an absolute-offset random-access
#   buffer: each public sensor entry point (CameraSensor, ImageSensor,
#   make_index_and_metadata, decoder ``open``) positions the stream at offset
#   0 before handing it to PyAV / PIL, and the underlying decoder performs
#   absolute seeks within the stream as it pleases. The library makes no
#   promises about the stream's position before, during, or after use, and
#   does not restore the caller's position on exit. Concurrent use of one
#   stream by multiple readers (or overlapping sensor calls) is unsupported.
#
# The library is intentionally backend-agnostic and never accepts URIs. To
# feed data from S3, Azure Blob, GCS, or any other remote store, callers
# wrap the remote stream as a ``BinaryIO`` (e.g. via boto3's
# ``StreamingBody``, ``azure-storage-blob``'s download stream, or
# ``smart_open.open(uri, "rb")``) and hand it in. If the remote data lives
# at a non-zero offset within a larger buffer, the caller is responsible
# for wrapping it in a slicing adapter that presents only the relevant
# bytes — the library treats absolute offset 0 as the data origin.
type DataSource = Path | bytes | io.BufferedIOBase


class TimestampSource(enum.Enum):
    """Which clock a sensor's reported timeline came from.

    ``SENSOR`` -- the recording's own sensor times, supplied by the caller from
    whatever the recorder wrote beside the media. These are what every timing question
    is actually about.

    ``CONTAINER_PTS`` -- the container's presentation timestamps, which reflect sensor time
    only when the file was muxed at a variable frame rate. A constant-frame-rate mux
    discards the sensor times and regenerates them on a fixed grid restarting at ``0``
    per file, so a timeline from this source can be a metronome that says nothing about
    what the hardware did.

    Reported so a reader can tell those two apart, because their timing numbers do not
    mean the same thing and nothing in the numbers themselves gives it away.
    """

    SENSOR = "sensor"
    CONTAINER_PTS = "container_pts"


class VideoIndexCreationMethod(enum.Enum):
    """How packet-level metadata is collected when building a video index.

    ``FROM_HEADER`` reads the stream's index entries parsed from the container
    header (fast), but the entry timestamps are decode timestamps (DTS). For
    B-frame video DTS != PTS, so a header-built index is sorted by DTS and does
    not match the true presentation order — producing wrong decode plans and
    spurious header-vs-demux mismatches.

    ``FULL_DEMUX`` walks every packet via demux (slow, I/O heavy) and reads each
    packet's PTS, so it is always correct.

    ``AUTO`` (default) reads ``FROM_HEADER`` when the stream has no B-frames
    (``codec_context.has_b_frames`` is 0; DTS == PTS, so the fast path is exact)
    and otherwise falls back to ``FULL_DEMUX``.
    """

    FROM_HEADER = "from_header"
    FULL_DEMUX = "full_demux"
    AUTO = "auto"
