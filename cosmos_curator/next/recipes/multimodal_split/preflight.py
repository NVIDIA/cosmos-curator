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


"""Prove a run can read its input and write its output, before it starts one.

Cheap, and in the driver. Credentials that are missing or wrong otherwise
surface as a failure per session inside Ray: a thousand sessions all recording
the same cause, a run that reports zero clips and exits successfully, and a
report tree to read before anyone can say what went wrong.

Nothing here branches on a scheme. ``get_storage_client`` answers ``None`` for a
local path and a client for a remote one, which is the same discriminator the
rest of this recipe uses, and S3 and Azure offer the operations below under one
signature.
"""

import uuid

from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_storage_client, path_to_prefix
from cosmos_curator.next.recipes.multimodal_split.uris import slashed
from cosmos_curator.next.utils.local_uri import local_path

# Says what it is and whose it is, so an operator who finds one knows it is ours
# and safe to remove. Unique per attempt, because two runs may preflight the
# same prefix at once and neither may delete the other's.
_PROBE_PREFIX = ".multimodal-split-preflight"


def check_readable(location: str, *, storage_profile: str = "default") -> None:
    """Raise unless *location* can be read with the credentials this run will use.

    Listing one object rather than stating the prefix: object storage has no
    prefix to stat, and a single-page list proves the two things that matter --
    that credentials resolve and that the prefix is reachable.

    Whether anything is *in* it is discovery's question, not this one.
    """
    try:
        # Inside the try: resolving a client is where a missing credential file
        # is discovered, and that error says nothing about which prefix wanted it.
        client = get_storage_client(location, profile_name=storage_profile)
        if client is None:
            if not local_path(location).is_dir():
                msg = f"Input prefix {location} is not a readable directory"
                raise NotADirectoryError(msg)  # noqa: TRY301 -- reported as the check's own failure
        else:
            # Trailing slash: object storage matches a prefix as a raw string,
            # so listing "sessions/foo" also finds "sessions/foo-bar/...", and a
            # prefix that holds nothing would pass on a sibling's contents.
            client.list_recursive_directory(path_to_prefix(slashed(location)), limit=1)
    except Exception as error:
        msg = f"Cannot read the input prefix {location}: {type(error).__name__}: {error}"
        raise RuntimeError(msg) from error


def check_writable(location: str, *, storage_profile: str = "default") -> None:
    """Raise unless *location* can be written with the credentials this run will use.

    Writes a probe object and removes it, because permission to write is not
    something object storage will answer any other way. A failure to remove it
    is logged rather than raised: by then the thing this set out to prove is
    proven, and refusing the run over a leftover byte would be the wrong trade.
    """
    probe_name = f"{_PROBE_PREFIX}-{uuid.uuid4().hex}"
    probe = f"{location.rstrip('/')}/{probe_name}"
    client = None
    try:
        client = get_storage_client(location, profile_name=storage_profile, can_overwrite=True, can_delete=True)
        if client is None:
            path = local_path(location) / probe_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        else:
            client.upload_bytes(path_to_prefix(probe), b"")
    except Exception as error:
        msg = f"Cannot write to the output path {location}: {type(error).__name__}: {error}"
        raise RuntimeError(msg) from error

    try:
        if client is None:
            (local_path(location) / probe_name).unlink()
        else:
            client.delete_object(path_to_prefix(probe))
    except Exception as error:  # noqa: BLE001 -- the write already proved what this checks
        logger.warning("Left a preflight probe behind at {}: {!r}", probe, error)
