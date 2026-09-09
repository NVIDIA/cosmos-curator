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
from cosmos_curator.next.utils.identity import canonical_digest

# Bump when the meaning of written media changes in a way the settings do not
# express, so corrected clips re-key instead of overwriting the old ones.
#
# 2: the identity stopped covering settings and became geometry alone. The
# digest inputs changed, so a version left at 1 would name two different
# identity schemes -- ids from either would be version 1 and indistinguishable.
MEDIA_CONTRACT_VERSION = 2


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


def make_clip_id(source_id: str, span: Span) -> str:
    """Identify one aligned clip independently of run order, so a retry addresses the same object.

    Takes the session's ``source_id`` rather than its name, so the same
    directory name under two input prefixes does not name the same clip; see
    :func:`make_source_id`.

    A clip is one session and one span of it, and its sensors are files inside it
    rather than clips of their own. So the identity carries no sensor key: it
    names the directory every sensor writes into, and the file name tells two
    views apart. That is ``robot_action_split``'s ``span_group_id`` -- the moment,
    shared across views -- and like it, this carries geometry alone.

    Settings are deliberately absent. The rate a clip was sampled at, the encoder
    it was written with, which cameras a rig had: all of them change the bytes,
    and none of them change *which moment* this is. Folding them in meant
    re-keying every clip in a dataset to change a bitrate, scattering two
    encodings of one moment under two names.

    What that gave up is a side effect the identity was never the right place to
    provide: while settings were in the key, a rerun under new ones could not
    land on the old media because it addressed a different object. It can now,
    and the recipe has no overwrite contract to replace it -- ``output`` carries
    a path and nothing else, and ``write_media`` replaces an object in place. So
    rerunning into a populated path rewrites what it produces and leaves behind
    whatever it no longer produces, such as a camera dropped from the config.
    Making that safe is an overwrite-and-cleanup contract on ``output``, not a
    property to recover by putting settings back in the key.

    ``MEDIA_CONTRACT_VERSION`` is the lever for when the meaning of a clip
    changes rather than its settings -- a fix to which frame a timestamp selects,
    say. Bumping it re-keys every clip, so corrected media lands beside the old
    rather than overwriting it in place.
    """
    return canonical_digest(
        {
            "source_id": source_id,
            "start_ns": span.start_ns,
            "end_ns": span.end_ns,
            "media_contract_version": MEDIA_CONTRACT_VERSION,
        }
    )
