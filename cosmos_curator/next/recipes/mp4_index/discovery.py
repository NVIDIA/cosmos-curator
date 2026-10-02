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

"""Discover candidate MP4s beneath one source prefix, or take an explicit list as-is."""

import fnmatch
import pathlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import get_files_relative, get_storage_client, is_remote_path
from cosmos_curator.next.recipes.mp4_index.config import Mp4IndexInputConfig

#: Keeps concurrent per-session listing calls under the host's own rate limit.
_S3_LISTING_CONCURRENCY = 8


@dataclass(frozen=True)
class VideoSource:
    """One discovered candidate video, before anything about it has been read."""

    media_location: str


def _list_local_relative_paths(path_prefix: str) -> list[str]:
    root = pathlib.Path(path_prefix)
    return [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]


def _list_s3_relative_paths_concurrently(path_prefix: str, *, client: S3Client) -> list[str]:
    """List every object beneath *path_prefix*, walking each top-level "directory" concurrently."""
    prefix = S3Prefix(path_prefix)
    try:
        session_names = client.list_child_prefixes(prefix)
    except FileNotFoundError:
        return []
    if not session_names:
        return get_files_relative(path_prefix, client=client)

    base = str(prefix).rstrip("/")
    with ThreadPoolExecutor(max_workers=min(_S3_LISTING_CONCURRENCY, len(session_names))) as pool:
        batches = pool.map(lambda name: get_files_relative(f"{base}/{name}", client=client), session_names)
        return [f"{name}/{relative}" for name, batch in zip(session_names, batches, strict=True) for relative in batch]


def _list_remote_relative_paths(path_prefix: str, *, client: StorageClient) -> list[str]:
    """List every object beneath *path_prefix*: concurrent subtree walk for an ``S3Client``, else a full listing."""
    if isinstance(client, S3Client):
        return _list_s3_relative_paths_concurrently(path_prefix, client=client)
    return get_files_relative(path_prefix, client=client)


def discover_videos(
    *,
    path_prefix: str,
    globs: list[str],
    exclude_globs: list[str] | None = None,
    limit: int | None = None,
    storage_profile: str = "default",
) -> list[VideoSource]:
    """List every object beneath *path_prefix* matching *globs* and not *exclude_globs*.

    Args:
        path_prefix: local directory or ``s3://`` prefix holding candidate videos.
        globs: ``fnmatch``'s ``*`` crosses ``/``, so a pattern like ``"*/unit-*/*.mp4"`` already
            matches at any depth with no special-casing for a "session" level.
        exclude_globs: drops an already-matched relative path if any of these also matches it.
        limit: optional cap applied after matching, so a smoke run can bound cost.
        storage_profile: profile name used to build the storage client for a remote *path_prefix*.

    Returns:
        Sources sorted by ``media_location``, for deterministic ordering across runs.

    """
    if is_remote_path(path_prefix):
        client = get_storage_client(path_prefix, profile_name=storage_profile)
        assert client is not None  # is_remote_path already confirmed a backend get_storage_client resolves
        relative_paths = _list_remote_relative_paths(path_prefix, client=client)
    else:
        relative_paths = _list_local_relative_paths(path_prefix)

    sources: list[VideoSource] = []
    for relative in relative_paths:
        if not any(fnmatch.fnmatch(relative, pattern) for pattern in globs):
            continue
        if exclude_globs and any(fnmatch.fnmatch(relative, pattern) for pattern in exclude_globs):
            continue
        media_location = f"{path_prefix.rstrip('/')}/{relative}"
        sources.append(VideoSource(media_location=media_location))

    sources.sort(key=lambda source: source.media_location)
    if limit is not None:
        sources = sources[:limit]
    return sources


def resolve_input_selection(
    input_config: Mp4IndexInputConfig, *, storage_profile: str = "default"
) -> list[VideoSource]:
    """Realize one source list from either input mode -- the top-level entry point callers use."""
    if input_config.uris is not None:
        return [VideoSource(media_location=uri) for uri in sorted(set(input_config.uris))]

    assert input_config.path_prefix is not None  # guarded by the caller's own config validation
    return discover_videos(
        path_prefix=input_config.path_prefix,
        globs=list(input_config.globs),
        exclude_globs=None if input_config.exclude_globs is None else list(input_config.exclude_globs),
        limit=input_config.limit,
        storage_profile=storage_profile,
    )
