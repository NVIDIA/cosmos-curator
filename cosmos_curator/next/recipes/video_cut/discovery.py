# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover identified source work items before Ray execution or output creation."""

import pyarrow as pa

from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.next.recipes.video_cut.config import InputConfig, ResolvedVideoCutConfig
from cosmos_curator.next.recipes.video_cut.contracts import INPUT_SCHEMA, make_source_id
from cosmos_curator.next.utils.source_objects import s3_client
from cosmos_curator.next.utils.uris import normalize_s3_uri


def discover_items(config: ResolvedVideoCutConfig) -> pa.Table:
    """Discover normalized source URIs and identify them in the append contract's input schema."""
    uris = _resolve_input_uris(config.input, storage_profile=config.execution.storage_profile)
    return pa.Table.from_pylist(
        [{"source_id": make_source_id(uri), "source_uri": uri} for uri in uris], schema=INPUT_SCHEMA
    )


def _resolve_input_uris(config: InputConfig, *, storage_profile: str) -> tuple[str, ...]:
    """Select explicit objects or all MP4 descendants of a directory-like S3 root."""
    if config.uris is not None:
        return config.uris
    if config.root_uri is None:  # Guarded by config validation.
        msg = "input must set uris or root_uri"
        raise ValueError(msg)
    root = S3Prefix(config.root_uri)
    # A directory boundary prevents root 'raw' from also selecting 'raw-old'.
    prefix = f"{root.prefix}/" if root.prefix else ""
    listing_root = S3Prefix(f"s3://{root.bucket}/{prefix}")
    sources = s3_client(storage_profile).list_recursive_with_suffixes(listing_root, (".mp4",))
    uris = set()
    for source in sources:
        if not isinstance(source, S3Prefix):
            msg = f"S3 discovery returned an unexpected object type under {listing_root.path}: {type(source).__name__}"
            raise TypeError(msg)
        if source.bucket != root.bucket or not source.prefix.startswith(prefix):
            msg = f"S3 discovery returned an object outside {listing_root.path}: {source.path}"
            raise ValueError(msg)
        uris.add(normalize_s3_uri(source.path))
    return tuple(sorted(uris))
