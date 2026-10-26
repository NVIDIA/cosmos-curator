# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credentialed S3 access to source videos and the failure policy both cutting methods share.

Input storage is stable, so a missing or archived object is a terminal error;
other storage failures propagate. Each method owns its source access strategy.
"""

from functools import lru_cache

from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix, get_s3_client_config
from cosmos_curator.next.recipes.video_cut.contracts import error_details

# Object states that stable storage never changes: the key is missing or the object is archived.
_UNREADABLE_OBJECT_CODES = frozenset({"NoSuchKey", "InvalidObjectState"})
# A HEAD request's bare 404, which does not say whether the key or the bucket is missing.
_HEAD_NOT_FOUND_CODES = frozenset({"404", "NotFound"})


@lru_cache(maxsize=8)
def s3_client(profile: str) -> S3Client:
    """Reuse storage configuration within a worker process."""
    return S3Client(get_s3_client_config(profile))


def unreadable_source_error(exc: ClientError, source_uri: str, *, storage_profile: str) -> dict[str, str] | None:
    """Classify a source that stays unreadable as terminal; return ``None`` for failures that must propagate.

    A missing key or an archived object is a stable property of the source.
    Missing buckets, credentials, permissions, endpoints, and throttling describe
    the environment or a transient condition, so they propagate and leave the
    source retryable. A bare 404 is a missing key only when the bucket exists.
    """
    code = str(exc.response.get("Error", {}).get("Code"))
    missing_key = code in _HEAD_NOT_FOUND_CODES and _bucket_exists(source_uri, storage_profile=storage_profile)
    if missing_key or code in _UNREADABLE_OBJECT_CODES:
        return error_details("source-read", f"S3 source unavailable: {code}")
    return None


def _bucket_exists(source_uri: str, *, storage_profile: str) -> bool:
    """Confirm the source's bucket is reachable; any failure leaves a bare 404 unexplained."""
    try:
        s3_client(storage_profile).s3.head_bucket(Bucket=S3Prefix(source_uri).bucket)
    except ClientError:
        return False
    return True
