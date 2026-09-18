# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 identity and cross-backend validation preserved from the retired splitter."""

import pytest

from cosmos_curator.core.utils.storage.s3_validation import MAX_S3_KEY_LENGTH_BYTES
from cosmos_curator.next.utils.uris import normalize_s3_uri


@pytest.mark.parametrize("key", ["a.mp4", "/a.mp4", "a.mp4/", "a%2Fb//c.mp4", "run=2026-09-18/a.mp4"])
def test_normalization_preserves_literal_object_keys(key: str) -> None:
    """URI normalization must not select a different immutable source object."""
    assert normalize_s3_uri(f"S3://example-bucket/{key}") == f"s3://example-bucket/{key}"


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("s3://example-bucket", "s3://example-bucket/"),
        ("s3://example-bucket/", "s3://example-bucket/"),
        ("s3://example-bucket/prefix/", "s3://example-bucket/prefix"),
    ],
)
def test_prefix_normalization_retains_bucket_root(location: str, expected: str) -> None:
    """Caption configs retain the bucket separator when normalizing root locations."""
    assert normalize_s3_uri(location, strip_trailing_slash=True) == expected


@pytest.mark.parametrize("bucket", ["example_bucket", f"a{'_' * 253}z"])
def test_s3_compatible_bucket_names(bucket: str) -> None:
    """Ceph and SwiftStack buckets may contain underscores and reach 255 characters."""
    assert normalize_s3_uri(f"s3://{bucket}/a.mp4") == f"s3://{bucket}/a.mp4"


@pytest.mark.parametrize("bucket", ["ab", f"a{'b' * 254}z"])
def test_invalid_bucket_lengths(bucket: str) -> None:
    """Configuration uses the same bucket limits as storage access."""
    with pytest.raises(ValueError, match="Invalid S3 bucket name"):
        normalize_s3_uri(f"s3://{bucket}/a.mp4")


@pytest.mark.parametrize("character", ["a", "é"])
def test_object_key_limit_is_measured_in_utf8_bytes(character: str) -> None:
    """Preserve legal keys at the boundary and reject those beyond S3's byte limit."""
    key = character * (MAX_S3_KEY_LENGTH_BYTES // len(character.encode("utf-8")))
    assert normalize_s3_uri(f"s3://example-bucket/{key}") == f"s3://example-bucket/{key}"
    with pytest.raises(ValueError, match="Invalid S3 object key"):
        normalize_s3_uri(f"s3://example-bucket/{key}{character}")


@pytest.mark.parametrize(
    "location",
    [
        "https://example-bucket/a.mp4",
        "s3://example-bucket/a.mp4?signature=secret",
        "s3://example-bucket/a.mp4?",
        "s3://example-bucket/a.mp4#part",
        "s3://example-bucket/a.mp4#",
        " s3://example-bucket/a.mp4",
        "s3://example-bucket/a\n.mp4",
    ],
)
def test_decorated_locations_are_rejected(location: str) -> None:
    """Reject URI decorations instead of silently changing object identity."""
    with pytest.raises(ValueError, match="Expected an undecorated S3 URI"):
        normalize_s3_uri(location)
