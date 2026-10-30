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

"""Discover the video streams that make up a single data-integrity session.

A session is one recording -- for example a ``clips/<uuid>/`` prefix whose
per-camera videos live under ``recorder-XX/`` subprefixes. ``discover_streams``
finds those video files under a session path (a local directory, a cloud prefix,
or a single file) so the runner can open each as a ``CameraSensor``. Non-video
session files (rig calibration JSON, upload markers) are ignored by the
extension filter.

Discovery recurses, which is why it also takes ``stream_globs``: a session prefix
that holds copies of its cameras alongside the originals -- a ``__original/`` or
``triage/`` sibling of the ``recorder-XX/`` directories -- would otherwise measure
each camera several times and tell the session-grain metrics that one rig had
twenty sensors. Naming the subprefixes that hold the real streams is what lets a
session stay the whole recording without picking up its own copies.

Multi-session grouping (running many ``<uuid>`` sessions under a parent prefix)
is intentionally not handled here: it is a thin loop over the single-session
runner, not part of discovery.
"""

import pathlib

from loguru import logger

from cosmos_curator.core.sensors.data_integrity.engine import validate_non_negative_int
from cosmos_curator.core.utils.storage.storage_utils import is_remote_path
from cosmos_curator.core.utils.storage_cli import list_storage_objects, storage_object_exists

# Container suffixes CameraSensor can open (MP4 family + MKV). Lowercased for
# case-insensitive matching. MPEG-TS is intentionally excluded (rejected by the
# sensor library).
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".mkv")


def _is_video(name: str) -> bool:
    return name.lower().endswith(VIDEO_SUFFIXES)


def _matches_any(relative: str, patterns: tuple[str, ...]) -> bool:
    """Whether a session-relative path matches one of ``patterns``.

    ``full_match`` rather than ``match`` because a glob here has to be anchored at
    the session: ``match`` matches from the right, so ``recorder-*/*`` would accept
    ``__original/recorder-00/front.mp4`` -- exactly the copy the pattern exists to
    exclude. Segment-wise semantics come with it, so ``*`` stops at a ``/`` and
    ``**`` is the way to span subdirectories.
    """
    candidate = pathlib.PurePosixPath(relative)
    return any(candidate.full_match(pattern) for pattern in patterns)


def _session_base(session_path: str) -> str:
    """Return the prefix every stream of ``session_path`` begins with, delimiter included."""
    return f"{session_path.rstrip('/')}/"


def _keep_under_session(streams: list[str], *, session_path: str) -> list[str]:
    """Keep the streams that are genuinely under ``session_path``.

    Both backends list by key prefix and neither knows what a directory is, so asked
    for ``clips/abc123`` they also answer with ``clips/abc123-backup/front.mp4`` -- a
    different session, whose cameras would otherwise be counted among this one's
    sensors and so could invent a cross-sensor finding. The trailing delimiter is what
    separates the two, and it is added here because :func:`identity.normalize_source`
    has already dropped it: a prefix is written both ways by hand, and the store has
    to hash one session to one id however it was spelled.
    """
    base = _session_base(session_path)
    kept: list[str] = []
    for stream in streams:
        if not stream.startswith(base):
            logger.warning("discovered stream {} is not under session {}; skipping it", stream, session_path)
            continue
        kept.append(stream)
    return kept


def _keep_matching(streams: list[str], *, session_path: str, patterns: tuple[str, ...]) -> list[str]:
    """Keep the streams whose path below ``session_path`` matches ``patterns``."""
    base = _session_base(session_path)
    return [
        stream
        for stream in _keep_under_session(streams, session_path=session_path)
        if _matches_any(stream.removeprefix(base), patterns)
    ]


def discover_streams(  # noqa: PLR0913
    session_path: str,
    *,
    limit: int = 0,
    stream_globs: tuple[str, ...] = (),
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> list[str]:
    """Discover the video streams under a single session path.

    Args:
        session_path: A local directory, a local file, or an ``s3://`` / ``az://``
            prefix (typically one ``clips/<uuid>/``). A lone video file is also
            accepted and returned as a single-element list; a remote one is resolved
            as that exact object, so naming a video costs a metadata lookup rather
            than permission to list its bucket.
        limit: Maximum number of streams to return; ``0`` (default) means all.
            For cloud prefixes the cap is pushed into the listing so a large
            bucket is not paged in full -- unless ``stream_globs`` is set, since a
            listing capped before filtering would spend the cap on excluded keys.
            A session whose name is also the prefix of a sibling's can therefore
            come back short of the cap, the siblings having been counted by the
            listing and then dropped: a cap is a sampling convenience, and spending
            part of it is the cheaper mistake than measuring another session.
        stream_globs: Keep only streams whose path *below* ``session_path`` matches
            one of these globs; empty (default) keeps every video found. ``*`` does
            not cross ``/``, so ``recorder-*/*`` selects the cameras of every
            recorder and nothing under a sibling copy of them; ``**`` spans
            subdirectories. Ignored when ``session_path`` names a single file,
            which is already a selection of one.
        s3_profile_name: Optional AWS profile for ``s3://`` sources.
        azure_profile_name: Azure profile for ``az://`` sources.
        endpoint_url: Optional S3 endpoint override for S3-compatible stores
            (see ``s3_client.resolve_s3_endpoint_url``).

    Returns:
        Fully-qualified stream paths/URIs in sorted (discovery) order. Cloud URIs
        can be handed straight to ``storage_cli.open_storage_source``; local paths
        are absolute filesystem paths.

    Raises:
        FileNotFoundError: If a local ``session_path`` does not exist.
        ValueError: If ``limit`` is negative.

    """
    limit = validate_non_negative_int("limit", limit)
    if is_remote_path(session_path):
        if _is_video(session_path) and storage_object_exists(
            session_path,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
        ):
            # Resolved as the object it names rather than as a prefix to list under,
            # the same way the local branch below answers a file with itself. Listing
            # would spend a bucket-listing permission to come back with
            # ``clip.mp4.backup.mp4`` beside ``clip.mp4``, and a video nobody named
            # would be measured and stored against this session.
            return [session_path]
        listed = list_storage_objects(
            session_path,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
            limit=0 if stream_globs else limit,
            suffixes=VIDEO_SUFFIXES,
        )
        streams = sorted(listed)
        if not stream_globs:
            return _keep_under_session(streams, session_path=session_path)
        streams = _keep_matching(streams, session_path=session_path, patterns=stream_globs)
        return streams[:limit] if limit > 0 else streams

    path = pathlib.Path(session_path)
    if path.is_file():
        # Apply the same video-suffix filter as directory discovery so an explicit
        # non-video file (e.g. rig_meta.json) does not become a bogus stream.
        return [str(path)] if _is_video(path.name) else []
    if path.is_dir():
        candidates = (p for p in path.rglob("*") if p.is_file() and _is_video(p.name))
        if stream_globs:
            # relative_to rather than the string prefix the cloud branch strips: a
            # session named "." or "a/../b" yields walk results that do not start
            # with the path as spelled, and dropping them all would be silent.
            candidates = (p for p in candidates if _matches_any(p.relative_to(path).as_posix(), stream_globs))
        streams = sorted(str(p) for p in candidates)
        return streams[:limit] if limit > 0 else streams

    msg = f"session path does not exist: {session_path}"
    raise FileNotFoundError(msg)
