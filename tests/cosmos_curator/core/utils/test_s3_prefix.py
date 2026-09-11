# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for S3Prefix and is_s3path in cosmos_curator.core.utils.storage.s3_client."""

import pytest

from cosmos_curator.core.utils.storage.s3_client import (
    MAX_S3_KEY_LENGTH_BYTES,
    S3Prefix,
    is_s3path,
    validate_configured_s3_location,
)


def test_s3prefix_with_scheme() -> None:
    """Ensure S3Prefix correctly parses full S3 URIs (with scheme) into bucket, prefix, and path."""
    sp = S3Prefix("s3://bucket-name/path/to/object")
    assert sp.bucket == "bucket-name"
    assert sp.prefix == "path/to/object"
    assert sp.path == "s3://bucket-name/path/to/object"
    assert str(sp) == sp.path


def test_s3prefix_without_scheme() -> None:
    """Ensure S3Prefix correctly parses URIs without scheme into proper bucket and prefix."""
    sp = S3Prefix("bucket-name/path/to/object")
    assert sp.bucket == "bucket-name"
    assert sp.prefix == "path/to/object"
    assert sp.path == "s3://bucket-name/path/to/object"


def test_s3prefix_root_bucket() -> None:
    """Ensure S3Prefix handles bucket-only URIs, yielding empty prefix and a trailing slash in path."""
    sp = S3Prefix("s3://bucket-name")
    assert sp.bucket == "bucket-name"
    assert sp.prefix == ""
    # Expect trailing slash when prefix is empty
    assert sp.path == "s3://bucket-name/"


def test_s3prefix_with_hyphen_in_bucket_and_underscore_in_key() -> None:
    """Ensure hyphens in bucket names and underscores in object keys are valid."""
    # Hyphens allowed in bucket, underscores allowed in key
    sp = S3Prefix("s3://my-bucket-123/key_name-456")
    assert sp.bucket == "my-bucket-123"
    assert sp.prefix == "key_name-456"
    assert sp.path == "s3://my-bucket-123/key_name-456"


def test_s3prefix_with_spaces() -> None:
    """Ensure spaces in object keys are accepted while bucket names remain valid."""
    # Spaces allowed in object key but not in bucket
    sp = S3Prefix("s3://bucket-name/path with spaces")
    assert sp.bucket == "bucket-name"
    assert sp.prefix == "path with spaces"
    assert sp.path == "s3://bucket-name/path with spaces"


def test_s3prefix_with_commas() -> None:
    """Ensure commas in object keys are accepted."""
    sp = S3Prefix("s3://bucket-name/path,with,commas")
    assert sp.bucket == "bucket-name"
    assert sp.prefix == "path,with,commas"
    assert sp.path == "s3://bucket-name/path,with,commas"


def test_s3prefix_invalid_characters() -> None:
    """Ensure invalid characters (e.g., '?') in bucket or key raise a ValueError."""
    # '?' is not allowed in bucket or key
    with pytest.raises(ValueError, match=r"Invalid S3 bucket name"):
        S3Prefix("s3://bucket?/key")


def test_s3prefix_underscore_in_bucket_name() -> None:
    """Ensure S3-compatible stores may use underscores in bucket names."""
    sp = S3Prefix("s3://bucket_name/key")
    assert sp.bucket == "bucket_name"
    assert sp.prefix == "key"
    assert sp.path == "s3://bucket_name/key"


def test_s3prefix_accepts_a_bucket_at_the_cross_backend_length_limit() -> None:
    """Ceph RGW and SwiftStack bucket names may be as long as 255 characters."""
    bucket = f"a{'_' * 253}z"

    assert S3Prefix(f"s3://{bucket}/key").bucket == bucket


def test_s3prefix_rejects_a_bucket_over_the_cross_backend_length_limit() -> None:
    """The widened limit remains bounded so an accidentally repeated name fails early."""
    bucket = f"a{'b' * 254}z"

    with pytest.raises(ValueError, match=r"Invalid S3 bucket name"):
        S3Prefix(f"s3://{bucket}/key")


@pytest.mark.parametrize("bucket", ["ab", "_bucket", "bucket_", "bucket name", "Bucket"])
def test_s3prefix_rejects_obviously_malformed_bucket_names(bucket: str) -> None:
    """Keep a minimal typo guard without imposing AWS-specific DNS rules."""
    with pytest.raises(ValueError, match=r"Invalid S3 bucket name"):
        S3Prefix(f"s3://{bucket}/key")


def test_s3prefix_key_length_limit() -> None:
    """Ensure overly long object keys (exceeding 1024 bytes) raise a ValueError."""
    # Generate a key that exceeds 1024 bytes
    long_key = "a" * 1025
    uri = f"s3://validbucket/{long_key}"
    with pytest.raises(ValueError, match=r"Invalid S3 object key"):
        S3Prefix(uri)


def test_s3prefix_accepts_a_key_at_the_length_limit() -> None:
    """Ensure the limit is inclusive, so the longest legal key is not rejected."""
    key = "a" * MAX_S3_KEY_LENGTH_BYTES
    assert S3Prefix(f"s3://validbucket/{key}").prefix == key


def test_s3prefix_measures_the_key_limit_in_utf8_bytes_not_characters() -> None:
    """S3 spends its 1024-byte budget on the encoding, so multibyte keys run out sooner.

    A character count would accept this key and leave the store to reject it on first
    use, which is the failure this validation exists to move forward to config time.
    ``é`` encodes to two bytes, so half the limit in characters is all of it in bytes.
    """
    at_limit = "é" * (MAX_S3_KEY_LENGTH_BYTES // 2)
    assert S3Prefix(f"s3://validbucket/{at_limit}").prefix == at_limit

    over_limit = "é" * (MAX_S3_KEY_LENGTH_BYTES // 2 + 1)
    assert len(over_limit) < MAX_S3_KEY_LENGTH_BYTES

    with pytest.raises(ValueError, match=r"exceeds 1024 bytes \(1026\)"):
        S3Prefix(f"s3://validbucket/{over_limit}")


def test_s3prefix_accepts_a_hive_style_partition_key() -> None:
    """A ``key=value`` path segment is an ordinary S3 key and a common layout.

    This is the case that motivated dropping the character allowlist: partitioned
    drops are written as ``run=<date>/`` by convention, and rejecting them meant a
    valid prefix could not be configured at all.
    """
    sp = S3Prefix("s3://bucket-name/run=2026-08-01/clip.mp4")
    assert sp.prefix == "run=2026-08-01/clip.mp4"


@pytest.mark.parametrize("key", ["a+b/clip.mp4", "ts:2026/clip.mp4", "user@host/clip.mp4", "take(1)/clip.mp4"])
def test_s3prefix_accepts_other_legal_key_characters(key: str) -> None:
    """Characters the old allowlist omitted are legal in S3 and must round-trip.

    Parametrized rather than merged into one key so a future narrowing shows which
    character it broke, and to make the point that patching the allowlist one
    character at a time would not have finished the job.
    """
    assert S3Prefix(f"s3://bucket-name/{key}").prefix == key


@pytest.mark.parametrize("key", ["a*foo", "recordings/*/raw", "clip?.mp4"])
def test_s3prefix_represents_a_key_containing_a_literal_glob_character(key: str) -> None:
    """``*`` and ``?`` are legal in a key, and objects using them really exist.

    ``aws s3 ls`` will happily show an object named ``a*foo``. Both listing paths
    wrap the keys the store returns in an ``S3Prefix``, so refusing one here would
    abort an entire enumeration over a single object rather than reject a typo.
    """
    assert S3Prefix(f"s3://bucket-name/{key}").prefix == key


@pytest.mark.parametrize("key", ["a*foo", "recordings/*/raw", "clip?.mp4"])
def test_a_configured_location_rejects_glob_metacharacters(key: str) -> None:
    """A glob a human typed into a config is a typo, and no read path expands it.

    This is the other population: the run it would produce succeeds against S3 and
    matches zero objects, so config time is the only place the mistake is visible.
    """
    with pytest.raises(ValueError, match=r"Invalid S3 object key"):
        validate_configured_s3_location(f"s3://bucket-name/{key}")


def test_a_configured_location_still_runs_the_ordinary_prefix_validation() -> None:
    """The helper adds to ``S3Prefix``'s checks rather than replacing them."""
    with pytest.raises(ValueError, match=r"Invalid S3 bucket name"):
        validate_configured_s3_location("s3://UPPERCASE/key")


def test_s3prefix_allows_an_empty_key() -> None:
    """A bucket with a trailing slash and no key stays valid, as it was before."""
    assert S3Prefix("s3://bucket-name/").prefix == ""


def test_is_s3path_behaviour() -> None:
    """Ensure is_s3path correctly identifies valid and invalid S3 URIs."""
    assert is_s3path("s3://bucket/key")
    assert not is_s3path("http://example.com")
    assert not is_s3path("bucket/key")
    assert not is_s3path(None)
