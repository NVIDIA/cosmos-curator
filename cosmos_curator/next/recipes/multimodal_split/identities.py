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

"""Stable identities for the ``multimodal-split`` recipe.

What a clip is addressed by, and therefore what a rerun overwrites rather than
duplicates. Separate from the stage that writes it so that changing how a clip
is produced cannot quietly change where it lands.
"""

from typing import Any

from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.grids import clip_duration_ns
from cosmos_curator.next.utils.identity import canonical_digest

# Bump when the meaning of written media changes in a way the settings do not
# express, so corrected clips re-key instead of overwriting the old ones.
#
# 2: the identity stopped covering settings and became geometry alone. The
# digest inputs changed, so a version left at 1 would name two different
# identity schemes -- ids from either would be version 1 and indistinguishable.
# 3: the selected source sensor-timestamp sidecar was renamed from
# ``.observations.npy`` to ``.sensor_timestamps.npy``.
MEDIA_CONTRACT_VERSION = 3
EPISODE_IDENTITY_VERSION = 1
# 4: a camera's allow_pts_fallback joined the contract, because it decides
# whether the camera can produce output at all and which clock its timeline
# came from -- result-defining, not execution tuning, so a rerun that flips it
# must not reconcile against episodes planned under the old setting.
# 3: how an MCAP sensor is sampled joined the contract, because a sensor placed
# on the grid produces different bytes under each policy.
# 2: MCAP sensors joined the contract. A clip published under version 1 was cut
# from cameras alone, so it cannot be said to satisfy a config that also asks for
# a trajectory.
#
# Reconciliation refuses such a clip rather than re-planning it: an episode whose
# identity matches but whose contract does not is a conflict, and
# ``reconcile_planned_episode`` raises. So adding a sensor to a config that has
# already published needs a new output path, not a rerun into the old one.
SPLIT_CONTRACT_DIGEST_VERSION = 4
# 2: non-camera sensor artifacts joined the digest. An episode published under
# version 1 was its cameras and nothing else, so it cannot be compared byte for
# byte against one that also carries a trajectory.
# 3: camera artifact identity now uses the canonical sensor-timestamp field.
MATERIALIZATION_DIGEST_VERSION = 3


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


def make_episode_id(source_id: str, span: Span) -> str:
    """Identify one logical source interval independently of its rendering."""
    return canonical_digest(
        {
            "kind": "multimodal-av-episode",
            "episode_identity_version": EPISODE_IDENTITY_VERSION,
            "source_id": source_id,
            "start_ns": span.start_ns,
            "exclusive_end_ns": span.end_ns,
        }
    )


def make_split_contract_digest(config: ResolvedMultimodalSplitConfig) -> str:
    """Bind the settings that define the camera-media transformation."""
    return canonical_digest(
        {
            "kind": "multimodal-av-split-contract",
            "split_contract_digest_version": SPLIT_CONTRACT_DIGEST_VERSION,
            "media_contract_version": MEDIA_CONTRACT_VERSION,
            "clip": {
                "duration_ns": clip_duration_ns(config),
                "sampling_fps": config.clip.sampling_fps,
            },
            "cameras": [
                {
                    "camera_key": key,
                    "sensor_type": camera.sensor_type,
                    "allow_pts_fallback": camera.allow_pts_fallback,
                }
                for key, camera in sorted(config.cameras.items())
            ],
            # Every sensor that puts an artifact in a clip, so adding one makes
            # published clips stale rather than leaving them reconciling as
            # canonical against a contract they no longer satisfy. ``topic``
            # belongs here for the same reason: the same clip cut from a
            # different topic is different bytes under an unchanged identity.
            "mcap_sensors": [
                {
                    "sensor_key": key,
                    "sensor_type": sensor.sensor_type,
                    "topic": sensor.topic,
                    # How a sensor is placed on the grid decides what its clip
                    # holds, not merely how it was produced: the same span under
                    # nearest selection and under interpolation is different
                    # bytes. So is the bound, which decides which clips exist at
                    # all by refusing the ones it will not serve.
                    "sampling": sensor.sampling,
                    "sampling_bound_ns": sensor.sampling_bound_ns,
                }
                for key, sensor in sorted(config.mcap_sensors.items())
            ],
            "video": {
                "max_delta_ns": config.video.max_delta_ns,
                "encoder": {
                    "video_encoder": config.video.encoder.video_encoder,
                    "video_bitrate": config.video.encoder.video_bitrate,
                    "gop_size": config.video.encoder.gop_size,
                    "faststart": config.video.encoder.faststart,
                },
            },
        }
    )


def make_materialization_digest(  # noqa: PLR0913 -- every input is a normative digest field
    *,
    episode_id: str,
    split_clip_id: str,
    split_contract_digest: str,
    sampling_fps: int,
    row_count: int,
    camera_media: list[dict[str, Any]],
    sensor_media: list[dict[str, Any]] | None = None,
) -> str:
    """Bind an episode to the exact bytes of every artifact in its clip."""
    return canonical_digest(
        {
            "kind": "multimodal-av-episode-materialization",
            "materialization_digest_version": MATERIALIZATION_DIGEST_VERSION,
            "episode_id": episode_id,
            "split_clip_id": split_clip_id,
            "split_contract_digest": split_contract_digest,
            "sampling_fps": sampling_fps,
            "row_count": row_count,
            "camera_artifacts": [
                {
                    "camera_key": str(camera["camera_key"]),
                    "media_sha256": str(camera["media_sha256"]),
                    "timeline_sha256": str(camera["timeline_sha256"]),
                    "sensor_timestamps_sha256": str(camera["sensor_timestamps_sha256"]),
                }
                for camera in sorted(camera_media, key=lambda camera: str(camera["camera_key"]))
            ],
            # Present even when empty, so an episode that carries a trajectory
            # and one that was never asked for it cannot digest alike.
            "sensor_artifacts": [
                {
                    "sensor_key": str(sensor["sensor_key"]),
                    "sensor_type": str(sensor["sensor_type"]),
                    "media_sha256": str(sensor["media_sha256"]),
                }
                for sensor in sorted(sensor_media or [], key=lambda sensor: str(sensor["sensor_key"]))
            ],
        }
    )
