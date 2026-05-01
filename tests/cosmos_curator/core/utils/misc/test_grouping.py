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
"""Tests for grouping utility helpers."""

from cosmos_curator.core.utils.misc.grouping import (
    pairwise,
    split_by_chunk_size,
    split_into_n_chunks,
)


def test_split_by_chunk_size_includes_incomplete_chunk() -> None:
    """Last chunk is yielded even when undersized."""
    data = list(range(5))

    result = list(split_by_chunk_size(data, chunk_size=2))

    assert result == [[0, 1], [2, 3], [4]]


def test_split_by_chunk_size_respects_custom_size_func() -> None:
    """Custom size function controls chunk boundaries."""
    items = [
        ("alpha", 2),
        ("beta", 1),
        ("gamma", 2),
        ("delta", 1),
        ("epsilon", 1),
    ]

    result = list(split_by_chunk_size(items, chunk_size=3, custom_size_func=lambda item: item[1]))

    assert result == [
        [("alpha", 2), ("beta", 1)],
        [("gamma", 2), ("delta", 1)],
        [("epsilon", 1)],
    ]


def test_split_by_chunk_size_drops_incomplete_chunk_when_requested() -> None:
    """Optional flag allows dropping trailing chunk."""
    data = list(range(5))

    result = list(split_by_chunk_size(data, chunk_size=2, drop_incomplete_chunk=True))

    assert result == [[0, 1], [2, 3]]


def test_split_into_n_chunks_handles_short_iterables() -> None:
    """Iterables shorter than requested chunks return singletons."""
    data = [42, 43]

    result = list(split_into_n_chunks(data, num_chunks=5))

    assert result == [[42], [43]]


def test_split_into_n_chunks_distributes_items_evenly() -> None:
    """Workload is balanced across requested chunk count."""
    data = list(range(10))

    result = list(split_into_n_chunks(data, num_chunks=3))

    assert result == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]


def test_pairwise_returns_consecutive_pairs() -> None:
    """Yields adjacent pairs for sequences."""
    data = ["a", "b", "c", "d"]

    result = list(pairwise(data))

    assert result == [("a", "b"), ("b", "c"), ("c", "d")]


def test_pairwise_handles_short_iterables() -> None:
    """Short iterables produce no pairs."""
    assert list(pairwise([1])) == []
    assert list(pairwise([])) == []
