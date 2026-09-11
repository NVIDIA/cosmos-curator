# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credentialed S3 and Arrow filesystem access."""

from functools import lru_cache
from urllib.parse import urlsplit

import pyarrow.fs as pafs

from cosmos_curator.core.utils.storage.s3_client import S3Client, get_s3_client_config
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options


@lru_cache(maxsize=8)
def s3_client(profile: str) -> S3Client:
    """Reuse storage configuration within a worker process."""
    return S3Client(get_s3_client_config(profile))


def arrow_filesystem(location: str, profile: str) -> pafs.FileSystem:
    """Resolve the filesystem used for both Ray checkpoints and Parquet results."""
    if not location.startswith("s3://"):
        return pafs.LocalFileSystem()
    options = get_lance_storage_options(location, profile_name=profile) or {}
    endpoint = options.get("aws_endpoint")
    parsed = urlsplit(endpoint) if endpoint else None
    return pafs.S3FileSystem(
        access_key=options.get("aws_access_key_id"),
        secret_key=options.get("aws_secret_access_key"),
        session_token=options.get("aws_session_token"),
        region=options.get("aws_region"),
        endpoint_override=(parsed.netloc + parsed.path if parsed and parsed.scheme else endpoint),
        scheme=parsed.scheme if parsed and parsed.scheme else None,
    )


def filesystem_path(uri: str) -> str:
    """Strip S3's protocol for Arrow filesystem methods."""
    return uri.removeprefix("s3://")
