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

"""URI joining for the ``multimodal-split`` recipe.

Where a session's artifacts live, expressed as string joins. Nothing here opens
anything: a URI is a name, and resolving one to bytes is storage's job.
"""

_TIMESTAMPS_SIDECAR_SUFFIX = ".timestamps.npy"
_SENSOR_TIMESTAMPS_SUFFIX = ".sensor_timestamps.npy"
SESSIONS_PREFIX = "sessions"


def slashed(location: str) -> str:
    """Return *location* ending in one slash, so a listing cannot spill into a sibling.

    Object storage matches a prefix as a raw string: listing ``sessions/foo``
    also finds ``sessions/foo-bar/...``, so a prefix holding nothing would
    answer with a sibling's contents.
    """
    return location if location.endswith("/") else f"{location}/"


def timestamps_sidecar_uri(media_uri: str) -> str:
    """Return the ``.timestamps.npy`` sidecar beside *media_uri*.

    The filename does not identify its clock: a source camera uses it for sensor
    timestamps, while a clip produced by this recipe uses it for its alignment grid.
    """
    return f"{media_uri}{_TIMESTAMPS_SIDECAR_SUFFIX}"


def resolve_sensor_uri(session_uri: str, uri_pattern: str) -> str:
    """Join a camera's session-relative pattern onto a session URI."""
    return f"{session_uri.rstrip('/')}/{uri_pattern}"


def clip_media_uri(clip_dir: str, sensor_key: str, suffix: str = ".mp4") -> str:
    """Return one sensor's media inside a clip's directory.

    The clip id is the moment and the file name is the view, so this is where
    the two meet. A reader holding both can construct the path rather than
    listing the directory to find out what a view is called.

    The suffix is the container, which differs by sensor: a camera's clip is an
    MP4 and a trajectory's is an MCAP, and both sit under the same clip id.
    """
    return f"{clip_dir}/{sensor_key}{suffix}"


def sensor_timestamps_uri(clip_media_location: str) -> str:
    """Return the selected sensor-timestamps artifact beside a clip's media."""
    return f"{clip_media_location}{_SENSOR_TIMESTAMPS_SUFFIX}"


def clip_dir_uri(output_path: str, session_id: str, clip_id: str) -> str:
    """Return the directory holding one clip's artifacts.

    Built from the configured output path rather than from a written clip's
    recorded URI: the row carries that one normalized to a ``file://`` URI for
    local runs, and a writer given it would treat the scheme as a directory
    name. Shared with the split stage so a caption cannot land beside a clip
    that is somewhere else.
    """
    return f"{output_path.rstrip('/')}/{SESSIONS_PREFIX}/{session_id}/{clip_id}"
