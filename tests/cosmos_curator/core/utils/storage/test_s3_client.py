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
"""Tests for S3 client listing semantics."""

from typing import Any

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix


class _FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages
        self.last_paginate_kwargs: dict[str, object] | None = None

    def paginate(self, **kwargs: object) -> list[dict[str, Any]]:
        self.last_paginate_kwargs = kwargs
        return self._pages


class _FakeS3:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.paginator = _FakePaginator(pages)

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return self.paginator


def test_list_recursive_respects_limit_within_large_page() -> None:
    """Trim results to exact limit when a single page contains more entries than requested."""
    pages = [
        {
            "Contents": [
                {"Key": "root/a.mp4"},
                {"Key": "root/b.mp4"},
                {"Key": "root/c.mp4"},
            ]
        }
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=2)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]


def test_list_recursive_respects_limit_across_pages() -> None:
    """Trim to exact limit when overflow occurs after reading a subsequent page."""
    pages = [
        {
            "Contents": [
                {"Key": "root/a.mp4"},
            ]
        },
        {
            "Contents": [
                {"Key": "root/b.mp4"},
                {"Key": "root/c.mp4"},
            ]
        },
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=2)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]


def test_list_recursive_without_limit_returns_all_pages() -> None:
    """Return all objects when no limit is specified."""
    pages = [
        {"Contents": [{"Key": "root/a.mp4"}]},
        {"Contents": [{"Key": "root/b.mp4"}]},
    ]
    client = object.__new__(S3Client)
    client.s3 = _FakeS3(pages)

    results = client.list_recursive(S3Prefix("s3://bucket/root"), limit=0)
    assert len(results) == 2
    assert [item["Key"] for item in results] == ["root/a.mp4", "root/b.mp4"]
