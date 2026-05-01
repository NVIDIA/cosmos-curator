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

"""Unit tests for LazyData lifecycle and coerce() classmethod."""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.utils.data.lazy_data import LazyData


class TestResolve:
    """Tests for LazyData.resolve() in various states."""

    def test_resolve_inline(self) -> None:
        """Resolve returns inline value when set."""
        arr = np.array([1, 2, 3], dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)
        assert lazy.resolve() is arr

    def test_resolve_empty(self) -> None:
        """Resolve returns None when both value and ref are None."""
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData()
        assert lazy.resolve() is None

    def test_resolve_from_ref(self) -> None:
        """Resolve fetches from ref when value is None."""
        arr = np.array([10, 20], dtype=np.uint8)
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=mock_ref)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.get", return_value=arr) as mock_get:
            result = lazy.resolve()

        mock_get.assert_called_once_with(mock_ref)
        assert result is arr
        assert lazy.value is arr

    def test_resolve_caches_result(self) -> None:
        """Second resolve() returns cached value without calling ray.get again."""
        arr = np.array([5], dtype=np.uint8)
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=mock_ref)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.get", return_value=arr):
            lazy.resolve()
            result = lazy.resolve()

        assert result is arr


class TestStore:
    """Tests for LazyData.store() including hardened guard."""

    def test_store_pushes_to_plasma(self) -> None:
        """Store calls ray.put and clears value."""
        arr = np.array([1, 2, 3], dtype=np.uint8)
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.put", return_value=mock_ref) as mock_put:
            lazy.store()

        mock_put.assert_called_once_with(arr)
        assert lazy.value is None
        assert lazy.ref is mock_ref

    def test_store_noop_when_empty(self) -> None:
        """Store on empty LazyData is a no-op."""
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData()

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.put") as mock_put:
            lazy.store()

        mock_put.assert_not_called()

    def test_store_noop_when_already_stored(self) -> None:
        """Store when value is None (already stored) is a no-op."""
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=mock_ref)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.put") as mock_put:
            lazy.store()

        mock_put.assert_not_called()
        assert lazy.ref is mock_ref

    def test_store_materialized_clears_value_only(self) -> None:
        """Store in materialized state (both set) clears value, keeps ref, no ray.put."""
        arr = np.array([7], dtype=np.uint8)
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, ref=mock_ref)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.put") as mock_put:
            lazy.store()

        mock_put.assert_not_called()
        assert lazy.value is None
        assert lazy.ref is mock_ref


class TestRelease:
    """Tests for LazyData.release()."""

    def test_release_keeps_ref(self) -> None:
        """Release clears value but keeps ref."""
        mock_ref = MagicMock()
        arr = np.array([1], dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, ref=mock_ref)

        lazy.release()

        assert lazy.value is None
        assert lazy.ref is mock_ref


class TestDrop:
    """Tests for LazyData.drop()."""

    def test_drop_clears_both(self) -> None:
        """Drop clears value, ref, and resets nbytes to zero."""
        mock_ref = MagicMock()
        arr = np.array([1], dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, ref=mock_ref, nbytes=1024)

        lazy.drop()

        assert lazy.value is None
        assert lazy.ref is None
        assert lazy.nbytes == 0


class TestBool:
    """Tests for LazyData.__bool__()."""

    def test_bool_empty(self) -> None:
        """Empty LazyData is falsy."""
        assert not LazyData()

    def test_bool_inline(self) -> None:
        """LazyData with inline value is truthy."""
        assert LazyData(value=np.array([1], dtype=np.uint8))

    def test_bool_stored(self) -> None:
        """LazyData with only ref is truthy."""
        assert LazyData(ref=MagicMock())


class TestEqualityAndIdentity:
    """Tests for eq=False behavioral contract (identity-based comparison).

    LazyData uses @attrs.define(eq=False) because its ``value`` field can hold
    numpy arrays.  Attrs-generated __eq__ compares fields as tuples; for numpy
    arrays ``arr == arr`` returns a boolean *array*, not a scalar, which raises
    ``ValueError: The truth value of an array is ambiguous`` whenever Python
    tries to reduce it to a single bool (e.g. in tuple comparison, ``in``,
    dict lookup).

    With eq=False, Python falls back to object.__eq__ (identity),
    object.__ne__ (identity), and object.__hash__ (id-based).  These tests
    lock that guarantee across all comparison-dependent operations.
    """

    def test_eq_same_instance(self) -> None:
        """Same instance compares equal to itself (identity reflexivity)."""
        arr = np.arange(100, dtype=np.uint8)
        a: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)

        assert a == a  # noqa: PLR0124  -- intentional identity-reflexivity test

    def test_eq_different_instances_same_data(self) -> None:
        """Two instances with identical data are NOT equal (identity, not value)."""
        data = np.arange(64, dtype=np.uint8)
        a: LazyData[npt.NDArray[np.uint8]] = LazyData(value=data.copy())
        b: LazyData[npt.NDArray[np.uint8]] = LazyData(value=data.copy())

        assert a is not b
        assert a != b

    def test_eq_multi_element_array_no_value_error(self) -> None:
        """Regression: comparing LazyData with multi-element arrays must not raise.

        Before eq=False, attrs-generated __eq__ would call arr == arr on the
        value field, producing a boolean array.  Python would then raise
        ``ValueError: The truth value of an array is ambiguous`` when reducing
        it to a scalar for the tuple comparison.
        """
        arr = np.arange(1024, dtype=np.uint8)
        a: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)
        b: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr.copy())

        result = a == b  # must not raise ValueError
        assert result is False

    def test_ne_uses_identity(self) -> None:
        """!= uses identity, not element-wise array comparison."""
        arr = np.arange(256, dtype=np.uint8)
        a: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)
        b: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr.copy())

        assert a != b
        assert a == a  # noqa: PLR0124  -- intentional identity-reflexivity test

    def test_in_operator_uses_identity(self) -> None:
        """'in' operator on a list calls __eq__ per element -- must not crash."""
        arr = np.arange(512, dtype=np.uint8)
        a: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)
        b: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr.copy())

        assert a in [a, b]
        assert a not in [b]

    def test_hashable_as_dict_key_and_set_member(self) -> None:
        """LazyData is hashable via id-based object.__hash__."""
        arr = np.arange(128, dtype=np.uint8)
        a: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)
        b: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr.copy())

        assert {a: 1}[a] == 1
        assert len({a, b}) == 2
        assert hash(a) != hash(b)

    def test_repr_large_array_no_crash(self) -> None:
        """repr() completes without error even for large arrays."""
        arr = np.arange(10_000, dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, nbytes=arr.nbytes)

        result = repr(lazy)

        assert "LazyData" in result


class TestGetstate:
    """Tests for adaptive pickle behavior."""

    def test_getstate_inline(self) -> None:
        """Inline state pickles value, not ref."""
        arr = np.array([1, 2], dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)
        state = lazy.__getstate__()

        assert np.array_equal(state["value"], arr)
        assert state["ref"] is None

    def test_getstate_stored(self) -> None:
        """Stored state pickles ref only, excludes value."""
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=mock_ref)
        state = lazy.__getstate__()

        assert state["ref"] is mock_ref
        assert state["value"] is None

    def test_getstate_materialized_excludes_value(self) -> None:
        """Materialized state (both set) pickles ref only."""
        arr = np.array([3], dtype=np.uint8)
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, ref=mock_ref)
        state = lazy.__getstate__()

        assert state["ref"] is mock_ref
        assert state["value"] is None

    def test_setstate_restores(self) -> None:
        """Setstate restores from pickle state dict."""
        arr = np.array([4], dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData()
        lazy.__setstate__({"ref": None, "value": arr})

        assert np.array_equal(lazy.value, arr)  # type: ignore[arg-type]
        assert lazy.ref is None

    def test_roundtrip_pickle_inline(self) -> None:
        """Full pickle roundtrip preserves inline value."""
        arr = np.array([10, 20, 30], dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)

        restored = pickle.loads(pickle.dumps(lazy))  # noqa: S301

        assert np.array_equal(restored.value, arr)
        assert restored.ref is None


class TestCoerce:
    """Tests for LazyData.coerce() classmethod."""

    def test_coerce_from_none(self) -> None:
        """Coercing None produces empty LazyData."""
        result: LazyData[npt.NDArray[np.uint8]] = LazyData.coerce(None)

        assert result.value is None
        assert result.ref is None
        assert not result

    def test_coerce_from_array(self) -> None:
        """Coercing a numpy array wraps it in LazyData."""
        arr = np.array([1, 2, 3], dtype=np.uint8)
        result: LazyData[npt.NDArray[np.uint8]] = LazyData.coerce(arr)

        assert result.value is arr
        assert result.ref is None
        assert result

    def test_coerce_from_lazy_data(self) -> None:
        """Coercing a LazyData copies ref and value into new instance."""
        mock_ref = MagicMock()
        arr = np.array([5], dtype=np.uint8)
        original: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, ref=mock_ref)

        result: LazyData[npt.NDArray[np.uint8]] = LazyData.coerce(original)

        assert result is not original
        assert result.value is arr
        assert result.ref is mock_ref

    def test_coerce_from_bytes(self) -> None:
        """Coercing bytes auto-converts to numpy uint8 array."""
        raw = b"\x01\x02\x03\x04"
        result: LazyData[npt.NDArray[np.uint8]] = LazyData.coerce(raw)  # type: ignore[arg-type]

        assert result.value is not None
        assert isinstance(result.value, np.ndarray)
        assert result.value.dtype == np.uint8
        assert np.array_equal(result.value, np.array([1, 2, 3, 4], dtype=np.uint8))
        assert result.ref is None

    def test_coerce_captures_nbytes_from_array(self) -> None:
        """Coerce captures nbytes from numpy array."""
        arr = np.zeros(1024, dtype=np.uint8)
        result: LazyData[npt.NDArray[np.uint8]] = LazyData.coerce(arr)

        assert result.nbytes == 1024

    def test_coerce_captures_nbytes_from_bytes(self) -> None:
        """Coerce captures nbytes when auto-converting bytes to numpy."""
        raw = b"\x00" * 512
        result: LazyData[npt.NDArray[np.uint8]] = LazyData.coerce(raw)  # type: ignore[arg-type]

        assert result.nbytes == 512

    def test_coerce_preserves_nbytes_from_lazy_data(self) -> None:
        """Coerce preserves nbytes from source LazyData."""
        original: LazyData[npt.NDArray[np.uint8]] = LazyData(value=np.zeros(256, dtype=np.uint8), nbytes=256)
        result: LazyData[npt.NDArray[np.uint8]] = LazyData.coerce(original)

        assert result.nbytes == 256

    def test_coerce_none_has_zero_nbytes(self) -> None:
        """Coerce of None produces LazyData with nbytes=0."""
        result: LazyData[npt.NDArray[np.uint8]] = LazyData.coerce(None)

        assert result.nbytes == 0

    @pytest.mark.parametrize("value", [42, "hello", [1, 2, 3]])
    def test_coerce_from_other_types(self, value: object) -> None:
        """Coercing other types wraps them directly."""
        result: LazyData[object] = LazyData.coerce(value)

        assert result.value is value
        assert result.ref is None


class TestNbytes:
    """Tests for LazyData.nbytes metadata field lifecycle."""

    def test_empty_has_zero_nbytes(self) -> None:
        """Empty LazyData defaults to nbytes=0."""
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData()
        assert lazy.nbytes == 0

    def test_inline_constructor_sets_nbytes(self) -> None:
        """Direct construction with value and nbytes preserves both."""
        arr = np.zeros(2048, dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, nbytes=2048)

        assert lazy.nbytes == 2048

    def test_store_captures_nbytes(self) -> None:
        """store() captures nbytes from value before clearing it."""
        arr = np.zeros(4096, dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.put", return_value=MagicMock()):
            lazy.store()

        assert lazy.value is None
        assert lazy.nbytes == 4096

    def test_store_materialized_captures_nbytes(self) -> None:
        """store() in materialized state captures nbytes before clearing value."""
        arr = np.zeros(8192, dtype=np.uint8)
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, ref=mock_ref)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.put") as mock_put:
            lazy.store()

        mock_put.assert_not_called()
        assert lazy.value is None
        assert lazy.nbytes == 8192

    def test_store_preserves_existing_nbytes(self) -> None:
        """store() does not overwrite nbytes when already captured."""
        arr = np.zeros(100, dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, nbytes=100)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.put", return_value=MagicMock()):
            lazy.store()

        assert lazy.nbytes == 100

    def test_resolve_captures_nbytes_when_zero(self) -> None:
        """resolve() fills in nbytes if it was 0 (e.g. from legacy pickle)."""
        arr = np.zeros(512, dtype=np.uint8)
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=mock_ref, nbytes=0)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.get", return_value=arr):
            lazy.resolve()

        assert lazy.nbytes == 512

    def test_resolve_preserves_existing_nbytes(self) -> None:
        """resolve() does not overwrite nbytes when already set."""
        arr = np.zeros(512, dtype=np.uint8)
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=mock_ref, nbytes=1024)

        with patch("cosmos_curator.core.utils.data.lazy_data.ray.get", return_value=arr):
            lazy.resolve()

        assert lazy.nbytes == 1024

    def test_nbytes_survives_pickle_inline(self) -> None:
        """Verify nbytes persists through pickle roundtrip for inline state."""
        arr = np.zeros(768, dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, nbytes=768)

        restored = pickle.loads(pickle.dumps(lazy))  # noqa: S301

        assert restored.nbytes == 768

    def test_nbytes_survives_getstate_setstate(self) -> None:
        """Verify nbytes is preserved through __getstate__/__setstate__ roundtrip."""
        arr = np.zeros(2048, dtype=np.uint8)
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(value=arr, nbytes=2048)
        state = lazy.__getstate__()

        restored: LazyData[npt.NDArray[np.uint8]] = LazyData()
        restored.__setstate__(state)

        assert restored.nbytes == 2048

    def test_setstate_defaults_nbytes_when_missing(self) -> None:
        """__setstate__ defaults nbytes to 0 for legacy pickled data without nbytes."""
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData()
        lazy.__setstate__({"ref": None, "value": None})

        assert lazy.nbytes == 0

    def test_getstate_stored_includes_nbytes(self) -> None:
        """__getstate__ includes nbytes when in stored state."""
        mock_ref = MagicMock()
        lazy: LazyData[npt.NDArray[np.uint8]] = LazyData(ref=mock_ref, nbytes=4096)
        state = lazy.__getstate__()

        assert state["nbytes"] == 4096
        assert state["ref"] is mock_ref
        assert state["value"] is None
