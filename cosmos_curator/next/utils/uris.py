# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light location normalization for recipe configuration and discovery."""

import re
from urllib.parse import urlsplit

from cosmos_curator.next.utils.local_uri import local_media_path, uri_scheme

# Match S3Prefix's bucket policy, including underscores for S3-compatible stores.
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,61}[a-z0-9]$")


def normalize_s3_uri(value: str) -> str:
    """Preserve literal S3 keys, including percent escapes and repeated slashes."""
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or any(character in value for character in "\r\n\t?#")
        or parsed.scheme.lower() != "s3"
        or not _BUCKET.fullmatch(parsed.netloc)
        or parsed.query
        or parsed.fragment
    ):
        msg = f"Expected an undecorated S3 URI, got {value!r}"
        raise ValueError(msg)
    return f"s3://{parsed.netloc}/{parsed.path.removeprefix('/')}"


def normalize_location(value: str) -> str:
    """Normalize a local or S3 location without changing literal object keys."""
    if not value or value != value.strip():
        msg = "Locations must be non-empty and cannot contain surrounding whitespace"
        raise ValueError(msg)
    if uri_scheme(value) == "s3":
        return normalize_s3_uri(value).rstrip("/")
    return str(local_media_path(value).expanduser().resolve())
