# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The source-read failure policy both cutting methods share.

Input storage is stable, so a missing or archived object is a terminal error;
other storage failures propagate. Each method owns its source access strategy.
"""

from botocore.exceptions import ClientError

from cosmos_curator.next.recipes.video_cut.contracts import error_details
from cosmos_curator.next.utils.source_objects import is_unreadable_source


def unreadable_source_error(exc: ClientError, source_uri: str, *, storage_profile: str) -> dict[str, str] | None:
    """Classify a source that stays unreadable as terminal; return ``None`` for failures that must propagate.

    A missing key or an archived object is a stable property of the source.
    Missing buckets, credentials, permissions, endpoints, and throttling describe
    the environment or a transient condition, so they propagate and leave the
    source retryable. A bare 404 is a missing key only when the bucket exists.
    """
    if is_unreadable_source(exc, source_uri, storage_profile=storage_profile):
        code = str(exc.response.get("Error", {}).get("Code"))
        return error_details("source-read", f"S3 source unavailable: {code}")
    return None
