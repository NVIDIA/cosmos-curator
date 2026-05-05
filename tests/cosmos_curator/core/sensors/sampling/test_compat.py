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
"""Tests for decoder-utils-compatible sampling helpers."""

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.sampling.compat import make_decoder_utils_compat_grid
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid, SamplingWindow
from cosmos_curator.core.sensors.sampling.sampler import sample_window_indices

_NS_PER_SECOND = 1_000_000_000


def _decoder_utils_sample_timestamps_ns(
    start_ns: int,
    stop_ns: int,
    sample_rate_hz: float,
    *,
    endpoint: bool = True,
) -> npt.NDArray[np.int64]:
    start_s = np.float32(start_ns / _NS_PER_SECOND)
    stop_s = np.float32(stop_ns / _NS_PER_SECOND)
    sample_interval_s = 1.0 / sample_rate_hz
    sample_stop_s = stop_s
    if endpoint:
        sample_stop_s += sample_interval_s * 0.5

    sample_elements_s = np.arange(start_s, sample_stop_s, sample_interval_s, dtype=np.float32)
    if not endpoint and len(sample_elements_s) > 0 and np.isclose(sample_elements_s[-1], sample_stop_s):
        sample_elements_s = sample_elements_s[:-1]
    return (sample_elements_s * _NS_PER_SECOND).astype(np.int64)


def test_make_decoder_utils_compat_grid_stays_close_to_decoder_utils_endpoint_timeline() -> None:
    """Compatibility grid should approximate decoder_utils.sample_closest without float32 drift."""
    start_ns = 0
    stop_ns = 5 * _NS_PER_SECOND
    sample_rate_hz = 2.0
    sample_interval_ns = round(_NS_PER_SECOND / sample_rate_hz)
    expected_exclusive_end_ns = stop_ns + max(1, sample_interval_ns // 2)

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_decoder_utils_compat_grid(
        start_ns,
        stop_ns,
        sample_rate_hz,
    )

    assert got_start_ns == start_ns
    assert got_exclusive_end_ns == expected_exclusive_end_ns
    np.testing.assert_array_equal(
        got_timestamps_ns,
        np.arange(start_ns, expected_exclusive_end_ns, sample_interval_ns, dtype=np.int64),
    )
    np.testing.assert_allclose(
        got_timestamps_ns,
        _decoder_utils_sample_timestamps_ns(start_ns, stop_ns, sample_rate_hz),
        atol=1_024,
    )
    assert got_timestamps_ns.dtype == np.int64
    assert np.all(np.diff(got_timestamps_ns) > 0)


@pytest.mark.parametrize("start_ns", [33_033_033, 41_666_667])
def test_make_decoder_utils_compat_grid_output_constructs_sampling_grid_with_float32_drift(start_ns: int) -> None:
    """Float32 drift in the compatibility grid should not violate SamplingGrid invariants."""
    stop_ns = start_ns + 5 * _NS_PER_SECOND
    sample_rate_hz = 2.0

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_decoder_utils_compat_grid(
        start_ns,
        stop_ns,
        sample_rate_hz,
    )
    grid = SamplingGrid(
        start_ns=got_start_ns,
        exclusive_end_ns=got_exclusive_end_ns,
        timestamps_ns=got_timestamps_ns,
        stride_ns=max(1, got_exclusive_end_ns - got_start_ns),
        duration_ns=max(1, got_exclusive_end_ns - got_start_ns),
    )

    assert grid.start_ns == start_ns
    assert int(grid.timestamps_ns[0]) == start_ns


def test_make_decoder_utils_compat_grid_preserves_non_endpoint_stop_exclusion() -> None:
    """endpoint=False should preserve stop-exclusion behavior."""
    start_ns = 0
    stop_ns = 2 * _NS_PER_SECOND
    sample_rate_hz = 1.0

    got_start_ns, got_exclusive_end_ns, got_timestamps_ns = make_decoder_utils_compat_grid(
        start_ns,
        stop_ns,
        sample_rate_hz,
        endpoint=False,
    )

    assert got_start_ns == start_ns
    assert got_exclusive_end_ns == stop_ns
    np.testing.assert_array_equal(
        got_timestamps_ns,
        _decoder_utils_sample_timestamps_ns(start_ns, stop_ns, sample_rate_hz, endpoint=False),
    )


@pytest.mark.parametrize("sample_rate_hz", [0.0, -1.0])
def test_make_decoder_utils_compat_grid_raises_on_non_positive_sample_rate(sample_rate_hz: float) -> None:
    """Compatibility grid should reject non-positive sampling rates."""
    with pytest.raises(ValueError, match="sample_rate_hz must be greater than 0"):
        make_decoder_utils_compat_grid(0, _NS_PER_SECOND, sample_rate_hz)


def test_make_decoder_utils_compat_grid_raises_when_stop_precedes_start() -> None:
    """Compatibility grid should reject an inverted time span."""
    with pytest.raises(ValueError, match="stop_ns must be greater than or equal to start_ns"):
        make_decoder_utils_compat_grid(10, 0, 1.0)


def test_make_decoder_utils_compat_grid_rejects_duplicate_nanosecond_timestamps() -> None:
    """SamplingGrid timestamps must stay duplicate-free even when decoder_utils float32 samples do not."""
    with pytest.raises(ValueError, match="strictly increasing nanosecond grid"):
        make_decoder_utils_compat_grid(0, 10, 2_000_000_000.0)


def test_make_decoder_utils_compat_grid_preserves_supersampling_counts() -> None:
    """Duplicate source-frame selections should be represented as sampler counts, not duplicate grid timestamps."""
    start_ns = 0
    stop_ns = _NS_PER_SECOND
    _, exclusive_end_ns, timestamps_ns = make_decoder_utils_compat_grid(start_ns, stop_ns, 4.0)
    canonical = np.array([0, 500_000_000, 1_000_000_000], dtype=np.int64)
    window = SamplingWindow(start_ns=start_ns, exclusive_end_ns=exclusive_end_ns, timestamps_ns=timestamps_ns)

    indices, counts = sample_window_indices(canonical, window)

    np.testing.assert_array_equal(indices, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([2, 2, 1], dtype=np.int64))


def test_make_decoder_utils_compat_grid_samples_irregular_source_timestamps() -> None:
    """The compatibility grid should drive sampler choices that match decoder_utils-style nearest sampling."""
    start_ns = 0
    stop_ns = _NS_PER_SECOND
    _, exclusive_end_ns, timestamps_ns = make_decoder_utils_compat_grid(start_ns, stop_ns, 2.0)
    canonical = np.array([0, 333_000_000, 666_000_000, 1_000_000_000], dtype=np.int64)
    window = SamplingWindow(start_ns=start_ns, exclusive_end_ns=exclusive_end_ns, timestamps_ns=timestamps_ns)

    indices, counts = sample_window_indices(canonical, window)

    np.testing.assert_array_equal(indices, np.array([0, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1, 1], dtype=np.int64))
