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

"""Dependency-light canonical URI helpers shared by config and runtime code."""

from pathlib import PurePosixPath
from urllib.parse import urlsplit

from cosmos_curator.core.utils.storage.s3_validation import validate_s3_prefix


def normalize_s3_uri(location: str, *, strip_trailing_slash: bool = False) -> str:
    """Return one canonical ``s3://bucket/key`` URI and reject URI decorations."""
    if not location or location != location.strip():
        msg = "S3 locations must be non-empty and cannot have surrounding whitespace"
        raise ValueError(msg)

    parsed = urlsplit(location)
    if parsed.scheme.lower() != "s3" or not parsed.netloc:
        msg = f"Expected an s3:// URI, got {location!r}"
        raise ValueError(msg)
    if parsed.query or parsed.fragment:
        msg = f"S3 locations cannot contain a query or fragment: {location!r}"
        raise ValueError(msg)

    # Keys are taken verbatim, matching AWS CLI s3:// semantics. Percent-decoding
    # here would corrupt keys that legitimately contain '%' and would collapse two
    # distinct objects onto one source identity.
    key = parsed.path.removeprefix("/")
    validate_s3_prefix(parsed.netloc, key)
    normalized = f"s3://{parsed.netloc}/{key}"
    if strip_trailing_slash and key:
        normalized = normalized.rstrip("/")
    return normalized


def normalize_s3_mp4_uri(location: str) -> str:
    """Normalize an exact S3 object URI and require an MP4 object key."""
    normalized = normalize_s3_uri(location)
    key = urlsplit(normalized).path.removeprefix("/")
    if not key or PurePosixPath(key).suffix.lower() != ".mp4":
        msg = f"video-split inputs must be MP4 object URIs, got {location!r}"
        raise ValueError(msg)
    return normalized


def join_s3_uri(root: str, *parts: str) -> str:
    """Join relative object-key components beneath an S3 root."""
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"{root.rstrip('/')}/{suffix}" if suffix else root.rstrip("/")
