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

"""Unit tests for ref_resolver batch resolution utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.utils.data.lazy_data import LazyData
from cosmos_curator.core.utils.data.ref_resolver import batch_resolve, prefetch, resolve_as_ready


class TestPrefetch:
    """Tests for prefetch() non-blocking hint."""

    def test_prefetch_skips_inline(self) -> None:
        """Already-resolved items are not passed to ray.wait."""
        arr = np.array([1], dtype=np.uint8)
        items: list[LazyData[npt.NDArray[np.uint8]]] = [LazyData(value=arr), LazyData(value=arr)]

        with patch("cosmos_curator.core.utils.data.ref_resolver.ray.wait") as mock_wait:
            prefetch(items)

        mock_wait.assert_not_called()

    def test_prefetch_empty_sequence(self) -> None:
        """Empty input is a no-op."""
        with patch("cosmos_curator.core.utils.data.ref_resolver.ray.wait") as mock_wait:
            prefetch([])

        mock_wait.assert_not_called()

    def test_prefetch_skips_empty_items(self) -> None:
        """Items with neither value nor ref are skipped."""
        items: list[LazyData[npt.NDArray[np.uint8]]] = [LazyData(), LazyData()]

        with patch("cosmos_curator.core.utils.data.ref_resolver.ray.wait") as mock_wait:
            prefetch(items)

        mock_wait.assert_not_called()

    def test_prefetch_calls_ray_wait(self) -> None:
        """Items with refs get passed to ray.wait for pre-fetching."""
        ref_a = MagicMock(name="ref_a")
        ref_b = MagicMock(name="ref_b")
        items: list[LazyData[npt.NDArray[np.uint8]]] = [LazyData(ref=ref_a), LazyData(ref=ref_b)]

        with patch("cosmos_curator.core.utils.data.ref_resolver.ray.wait") as mock_wait:
            prefetch(items)

        mock_wait.assert_called_once_with([ref_a, ref_b], num_returns=2, timeout=0, fetch_local=True)

    def test_prefetch_mixed_items(self) -> None:
        """Only items with refs (and no value) are pre-fetched."""
        ref_a = MagicMock(name="ref_a")
        arr = np.array([1], dtype=np.uint8)
        items: list[LazyData[npt.NDArray[np.uint8]]] = [
            LazyData(value=arr),
            LazyData(ref=ref_a),
            LazyData(),
        ]

        with patch("cosmos_curator.core.utils.data.ref_resolver.ray.wait") as mock_wait:
            prefetch(items)

        mock_wait.assert_called_once_with([ref_a], num_returns=1, timeout=0, fetch_local=True)


class TestResolveAsReady:
    """Tests for resolve_as_ready() generator."""

    def test_inline_values_first(self) -> None:
        """Inline values are yielded before remote refs."""
        arr1 = np.array([1], dtype=np.uint8)
        arr2 = np.array([2], dtype=np.uint8)
        items: list[tuple[str, LazyData[npt.NDArray[np.uint8]]]] = [
            ("a", LazyData(value=arr1)),
            ("b", LazyData(value=arr2)),
        ]

        results = list(resolve_as_ready(items))  # type: ignore[arg-type]  # LazyData invariance: LazyData[T] vs LazyData[T | None]

        assert len(results) == 2
        assert results[0] == ("a", arr1)
        assert results[1] == ("b", arr2)

    def test_empty_yields_none(self) -> None:
        """Empty LazyData items yield (key, None) instead of being skipped."""
        items: list[tuple[str, LazyData[npt.NDArray[np.uint8]]]] = [
            ("a", LazyData()),
            ("b", LazyData()),
        ]

        results = list(resolve_as_ready(items))  # type: ignore[arg-type]  # LazyData invariance: LazyData[T] vs LazyData[T | None]

        assert len(results) == 2
        assert results[0] == ("a", None)
        assert results[1] == ("b", None)

    def test_mixed_inline_and_empty(self) -> None:
        """Mix of inline and empty items yields correct count."""
        arr = np.array([1], dtype=np.uint8)
        items: list[tuple[str, LazyData[npt.NDArray[np.uint8]]]] = [
            ("a", LazyData(value=arr)),
            ("b", LazyData()),
            ("c", LazyData(value=arr)),
        ]

        results = list(resolve_as_ready(items))  # type: ignore[arg-type]  # LazyData invariance: LazyData[T] vs LazyData[T | None]

        assert len(results) == 3
        assert results[0][0] == "a"
        assert results[0][1] is arr
        assert results[1] == ("b", None)
        assert results[2][0] == "c"

    def test_remote_refs_resolved(self) -> None:
        """Remote refs are resolved via ray.wait + ray.get."""
        arr = np.array([42], dtype=np.uint8)
        ref = MagicMock(name="ref")
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=ref)

        with (
            patch(
                "cosmos_curator.core.utils.data.ref_resolver.ray.wait",
                return_value=([ref], []),
            ),
            patch(
                "cosmos_curator.core.utils.data.lazy_data.ray.get",
                return_value=arr,
            ),
        ):
            results = list(resolve_as_ready([("k", lazy)]))  # type: ignore[list-item]  # LazyData invariance

        assert len(results) == 1
        assert results[0][0] == "k"
        assert results[0][1] is arr

    def test_duplicate_object_refs_all_yielded(self) -> None:
        """Multiple items sharing the same ObjectRef are all yielded."""
        arr = np.array([99], dtype=np.uint8)
        shared_ref = MagicMock(name="shared_ref")
        lazy_a: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=shared_ref)
        lazy_b: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=shared_ref)

        with (
            patch(
                "cosmos_curator.core.utils.data.ref_resolver.ray.wait",
                return_value=([shared_ref], []),
            ),
            patch(
                "cosmos_curator.core.utils.data.lazy_data.ray.get",
                return_value=arr,
            ),
        ):
            results = list(resolve_as_ready([("a", lazy_a), ("b", lazy_b)]))  # type: ignore[list-item]  # LazyData invariance

        assert len(results) == 2
        keys = {k for k, _ in results}
        assert keys == {"a", "b"}
        for _, val in results:
            assert np.array_equal(val, arr)  # type: ignore[arg-type]


class TestBatchResolve:
    """Tests for batch_resolve() with in-place mutation of LazyData instances."""

    def test_mutates_value_in_place(self) -> None:
        """batch_resolve populates .value on fetched LazyData instances."""
        ref = MagicMock(name="ref")
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=ref)
        fetched = np.array([1], dtype=np.uint8)

        with patch(
            "cosmos_curator.core.utils.data.ref_resolver.ray.get",
            return_value=[fetched],
        ):
            batch_resolve([lazy])

        assert lazy.value is fetched
        assert lazy.nbytes == 1

    def test_mixed_inline_remote_empty(self) -> None:
        """Mix of inline, remote, and empty returns correct list."""
        arr = np.array([10], dtype=np.uint8)
        remote_arr = np.array([20], dtype=np.uint8)
        ref = MagicMock(name="ref")
        ref_lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=ref)
        items: list[LazyData[npt.NDArray[np.uint8]]] = [
            LazyData(value=arr),
            ref_lazy,
            LazyData(),
        ]

        with patch(
            "cosmos_curator.core.utils.data.ref_resolver.ray.get",
            return_value=[remote_arr],
        ):
            results = batch_resolve(items)

        assert len(results) == 3
        assert results[0] is arr
        assert np.array_equal(results[1], remote_arr)  # type: ignore[arg-type]
        assert results[2] is None

    def test_all_inline(self) -> None:
        """All inline items resolved without ray.get."""
        arr1 = np.array([1], dtype=np.uint8)
        arr2 = np.array([2], dtype=np.uint8)
        items: list[LazyData[npt.NDArray[np.uint8]]] = [LazyData(value=arr1), LazyData(value=arr2)]

        with patch("cosmos_curator.core.utils.data.ref_resolver.ray.get") as mock_get:
            results = batch_resolve(items)

        mock_get.assert_not_called()
        assert results[0] is arr1
        assert results[1] is arr2

    def test_empty_list(self) -> None:
        """Empty input returns empty list."""
        results = batch_resolve([])

        assert results == []
