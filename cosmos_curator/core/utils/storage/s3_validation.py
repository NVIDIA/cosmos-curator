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

"""Dependency-light validation shared by S3 storage and configuration code."""

import re

# S3 accepts a key whose UTF-8 encoding is at most 1024 bytes and imposes no
# character allowlist beyond that.
MAX_S3_KEY_LENGTH_BYTES = 1024

# This deliberately targets the permissive overlap used by Ceph RGW, SwiftStack,
# and other S3-compatible stores: 3-255 characters with underscores allowed.
_BUCKET_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{1,253}[a-z0-9]")
_KEY_GLOB_CHARACTERS = ("*", "?")


def validate_s3_bucket_name(bucket: str) -> None:
    """Validate Curator's minimal cross-backend S3 bucket-name contract."""
    if not _BUCKET_NAME_PATTERN.fullmatch(bucket):
        error_msg = f"Invalid S3 bucket name: {bucket}"
        raise ValueError(error_msg)


def validate_s3_prefix(bucket: str, key: str) -> None:
    """Validate a bucket and key that may have come from an S3-compatible store."""
    validate_s3_bucket_name(bucket)

    # A character allowlist rejected legal keys such as Hive-style
    # ``run=2026-08-01/`` partitions. The limit is measured in UTF-8 bytes, so
    # multibyte characters consume more than one byte of the budget.
    key_length_bytes = len(key.encode("utf-8"))
    if key_length_bytes > MAX_S3_KEY_LENGTH_BYTES:
        error_msg = f"Invalid S3 object key: exceeds {MAX_S3_KEY_LENGTH_BYTES} bytes ({key_length_bytes})"
        raise ValueError(error_msg)


def validate_configured_s3_location(location: str) -> None:
    """Validate a human-authored S3 location, including likely glob typos."""
    bucket_and_key = location.removeprefix("s3://")
    bucket, _, key = bucket_and_key.partition("/")
    validate_s3_prefix(bucket, key)

    # Glob characters are legal in stored keys, but configured locations do not
    # expand them and would silently match no objects.
    if any(char in key for char in _KEY_GLOB_CHARACTERS):
        error_msg = f"Invalid S3 object key: {key} contains a glob metacharacter (* or ?), which is not expanded"
        raise ValueError(error_msg)
