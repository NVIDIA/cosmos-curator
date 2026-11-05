# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize source objects that stay unreadable, the only source storage failures recipes treat as terminal.

Recipes read immutable source datasets, so a missing or archived object is a
stable property of the source. Every other storage failure describes the
environment or a transient condition and must propagate, leaving the work
retryable. Recipes decide how a recognized failure becomes a row.
"""

from functools import lru_cache

from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix, get_s3_client_config

# Object states that stable storage never changes: the key is missing or the object is archived.
_UNREADABLE_OBJECT_CODES = frozenset({"NoSuchKey", "InvalidObjectState"})
# A HEAD request's bare 404, which does not say whether the key or the bucket is missing.
_HEAD_NOT_FOUND_CODES = frozenset({"404", "NotFound"})


@lru_cache(maxsize=8)
def s3_client(profile: str) -> S3Client:
    """Reuse storage configuration within a worker process."""
    return S3Client(get_s3_client_config(profile))


def is_unreadable_source(exc: BaseException, source_uri: str, *, storage_profile: str) -> bool:
    """Return whether ``exc`` shows the source object itself is missing or archived.

    A local file that does not exist is missing. On S3, a missing key or an
    archived object is terminal; a bare 404 means a missing key only when the
    bucket exists. Missing buckets, credentials, permissions, endpoints, and
    throttling are not terminal.
    """
    if isinstance(exc, FileNotFoundError):
        return True
    if not isinstance(exc, ClientError):
        return False
    code = str(exc.response.get("Error", {}).get("Code"))
    if code in _UNREADABLE_OBJECT_CODES:
        return True
    return code in _HEAD_NOT_FOUND_CODES and _bucket_exists(source_uri, storage_profile=storage_profile)


def _bucket_exists(source_uri: str, *, storage_profile: str) -> bool:
    """Confirm the source's bucket is reachable; any failure leaves a bare 404 unexplained."""
    try:
        s3_client(storage_profile).s3.head_bucket(Bucket=S3Prefix(source_uri).bucket)
    except ClientError:
        return False
    return True
