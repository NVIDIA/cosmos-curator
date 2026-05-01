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

from cosmos_curator.core.utils.storage.s3_client import S3Prefix, is_s3path


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
    """Ensure underscores in bucket names raise a ValueError."""
    # Underscore allowed in bucket name
    sp = S3Prefix("s3://bucket_name/key")
    assert sp.bucket == "bucket_name"
    assert sp.prefix == "key"
    assert sp.path == "s3://bucket_name/key"


def test_s3prefix_key_length_limit() -> None:
    """Ensure overly long object keys (exceeding 1024 characters) raise a ValueError."""
    # Generate a key that exceeds 1024 characters
    long_key = "a" * 1025
    uri = f"s3://validbucket/{long_key}"
    with pytest.raises(ValueError, match=r"Invalid S3 object key"):
        S3Prefix(uri)


def test_is_s3path_behaviour() -> None:
    """Ensure is_s3path correctly identifies valid and invalid S3 URIs."""
    assert is_s3path("s3://bucket/key")
    assert not is_s3path("http://example.com")
    assert not is_s3path("bucket/key")
    assert not is_s3path(None)
