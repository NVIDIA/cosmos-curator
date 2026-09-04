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

"""Clip identity for the ``multimodal-split`` recipe.

What a clip is addressed by, and therefore what a rerun overwrites rather than
duplicates. Separate from the stage that writes it so that changing how a clip
is produced cannot quietly change where it lands.
"""

from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.utils.identity import canonical_digest

# Bump when the meaning of written media changes in a way the settings do not
# express, so corrected clips re-key instead of overwriting the old ones.
MEDIA_CONTRACT_VERSION = 1


def make_source_id(session_uri: str) -> str:
    """Identify a session by where it is, not by what its directory is called.

    A session ID is one directory name under the input prefix, so two datasets
    can each hold a ``session-000`` and, named alone, would key the same clips
    into the same place. The URI is what tells them apart, and matching
    ``robot_action_split``, which digests a dataset root for the same reason,
    keeps one convention across the recipes rather than three.

    The cost is the contract's: a session that moves is a different source, so
    copying a dataset to another prefix re-keys every clip it holds.
    """
    return canonical_digest(session_uri.rstrip("/"))


def make_clip_id(source_id: str, span: Span, config: ResolvedMultimodalSplitConfig) -> str:
    """Identify one aligned clip independently of run order, so a retry addresses the same object.

    Takes the session's ``source_id`` rather than its name, so the same
    directory name under two input prefixes does not name the same clip; see
    :func:`make_source_id`.

    A clip is one session and one span of it, and its cameras are files inside
    it rather than clips of their own. So the identity carries no camera key: it
    names the directory every camera writes into, and the file name tells two
    views apart.

    It covers everything that decides what the written bytes *are*, because the
    identity is also the object key. That includes the whole camera map, since
    ``uri_pattern`` decides which artifact is read and ``max_delta_s`` decides
    which source frame serves each grid timestamp -- both change the pixels while
    still naming the same camera. The cost is that changing the camera set
    re-keys every clip, which is the honest outcome: a five-camera dataset is not
    the four-camera one with a file added.

    The ``execution`` section is deliberately absent. How much heap a task
    declares, how large a read is, how many cores planning takes -- none of them
    change a written byte, and folding them in would scatter identical media
    across different keys.

    ``MEDIA_CONTRACT_VERSION`` is the lever for everything the settings cannot
    express. When the meaning of the output changes -- a fix to which frame a
    timestamp selects, say -- bumping it re-keys every clip, so corrected media
    lands beside the old rather than overwriting it in place.
    """
    return canonical_digest(
        {
            "source_id": source_id,
            "start_ns": span.start_ns,
            "end_ns": span.end_ns,
            "output_fps": config.clip.output_fps,
            "encoder": config.encoder.model_dump(mode="json"),
            "cameras": {key: camera.uri_pattern for key, camera in config.sensors.cameras.items()},
            "max_delta_s": config.sensors.max_delta_s,
            "media_contract_version": MEDIA_CONTRACT_VERSION,
        }
    )
