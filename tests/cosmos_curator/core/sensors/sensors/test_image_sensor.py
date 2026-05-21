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

"""Tests for ``ImageSensor``."""

import io
import pathlib
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image as PILImage

from cosmos_curator.core.sensors.sampling.grid import SamplingGrid, SamplingWindow
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors import image_sensor as image_sensor_module
from cosmos_curator.core.sensors.sensors.image_sensor import ImageSensor, _resolve_sensor_timestamps


class _StaticGrid:
    def __init__(self, windows: list[npt.NDArray[np.int64]]) -> None:
        self._windows = []
        for t in windows:
            if len(t) == 0:
                self._windows.append(SamplingWindow(start_ns=0, exclusive_end_ns=0, timestamps_ns=t))
            else:
                self._windows.append(
                    SamplingWindow(
                        start_ns=int(t[0]),
                        exclusive_end_ns=int(t[-1]) + 1,
                        timestamps_ns=t,
                    )
                )

    def __iter__(self) -> Iterator[SamplingWindow]:
        return iter(self._windows)


class _StaticSpec:
    def __init__(self, windows: list[np.ndarray], policy: object = None) -> None:
        self.grid = _StaticGrid(windows)
        self.policy = policy


def _write_image(
    path: pathlib.Path,
    color: tuple[int, int, int],
    *,
    size: tuple[int, int] = (4, 3),
) -> None:
    PILImage.new("RGB", size, color).save(path)


def test_image_sensor_synthesizes_timestamps(tmp_path: pathlib.Path) -> None:
    """Missing timestamps should synthesize ``0..N-1``."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    sensor = ImageSensor([image_a, image_b])
    np.testing.assert_array_equal(sensor.sensor_timestamps_ns, np.array([0, 1], dtype=np.int64))


def test_resolve_sensor_timestamps_synthesizes_read_only_range() -> None:
    """The helper should synthesize a read-only ``0..N-1`` range when timestamps are omitted."""
    resolved = _resolve_sensor_timestamps(3, None)
    np.testing.assert_array_equal(resolved, np.array([0, 1, 2], dtype=np.int64))
    assert not resolved.flags.writeable


def test_resolve_sensor_timestamps_rejects_length_mismatch() -> None:
    """The helper should reject timestamp arrays whose length does not match the source count."""
    with pytest.raises(ValueError, match="must match sources length"):
        _resolve_sensor_timestamps(2, np.array([0], dtype=np.int64))


def test_image_sensor_rejects_empty_sources() -> None:
    """ImageSensor should require at least one source image."""
    with pytest.raises(ValueError, match="sources must be non-empty"):
        ImageSensor([])


def test_image_sensor_start_and_end_ns(tmp_path: pathlib.Path) -> None:
    """The sensor should expose the first and last sensor timestamps."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    sensor = ImageSensor([image_a, image_b], sensor_timestamps_ns=np.array([10, 30], dtype=np.int64))

    assert sensor.start_ns == 10
    assert sensor.end_ns == 30


def test_image_sensor_sample_uses_closest_timestamp(tmp_path: pathlib.Path) -> None:
    """Sampling should pick the image nearest to each reference timestamp in the window."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    sensor = ImageSensor([image_a, image_b], sensor_timestamps_ns=np.array([10, 30], dtype=np.int64))
    grid = SamplingGrid(
        start_ns=10,
        exclusive_end_ns=40,
        timestamps_ns=np.array([10, 29], dtype=np.int64),
        stride_ns=100,
        duration_ns=100,
    )

    sampled = next(sensor.sample(SamplingSpec(grid=grid)))

    np.testing.assert_array_equal(sampled.align_timestamps_ns, np.array([10, 29], dtype=np.int64))
    np.testing.assert_array_equal(sampled.sensor_timestamps_ns, np.array([10, 30], dtype=np.int64))
    assert sampled.frames.shape == (2, 3, 4, 3)
    assert tuple(sampled.frames[0, 0, 0]) == (255, 0, 0)
    assert tuple(sampled.frames[1, 0, 0]) == (0, 255, 0)


def test_image_sensor_sample_is_window_local(tmp_path: pathlib.Path) -> None:
    """Sampling should ignore a globally closer image that lies outside the current window."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    sensor = ImageSensor([image_a, image_b], sensor_timestamps_ns=np.array([10, 35], dtype=np.int64))
    grid = SamplingGrid(
        start_ns=29,
        exclusive_end_ns=30,
        timestamps_ns=np.array([29], dtype=np.int64),
        stride_ns=100,
        duration_ns=100,
    )

    sampled = next(sensor.sample(SamplingSpec(grid=grid)))

    assert sampled.align_timestamps_ns.shape == (0,)
    assert sampled.sensor_timestamps_ns.shape == (0,)
    assert sampled.frames.shape == (0, 3, 4, 3)


def test_image_sensor_sample_returns_empty_for_empty_window(tmp_path: pathlib.Path) -> None:
    """An empty window should yield the cached empty ImageData result."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))
    sampled_batches = list(sensor.sample(_StaticSpec([np.empty(0, dtype=np.int64)])))
    assert len(sampled_batches) == 1
    sampled = sampled_batches[0]

    assert sampled.align_timestamps_ns.shape == (0,)
    assert sampled.sensor_timestamps_ns.shape == (0,)
    assert sampled.frames.shape == (0, 3, 4, 3)


@pytest.mark.skip(reason="Sentinel-style exclusive-end marker removed; SamplingWindow handles boundaries explicitly")
def test_image_sensor_sample_returns_empty_when_active_grid_is_empty(tmp_path: pathlib.Path) -> None:
    """A window containing only its exclusive-end marker should yield empty data."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))
    sampled_batches = list(sensor.sample(_StaticSpec([np.array([10], dtype=np.int64)])))
    assert len(sampled_batches) == 1
    sampled = sampled_batches[0]

    assert sampled.align_timestamps_ns.shape == (0,)
    assert sampled.sensor_timestamps_ns.shape == (0,)
    assert sampled.frames.shape == (0, 3, 4, 3)


def test_image_sensor_sample_returns_empty_when_sampler_returns_no_indices(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the sampler returns no indices, the sensor should yield empty data."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    def _fake_sample_window_indices(
        sensor_timestamps_ns: np.ndarray,
        window: np.ndarray,
        *,
        policy: object,
        dedup: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        del sensor_timestamps_ns, window, policy, dedup
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    monkeypatch.setattr(image_sensor_module, "sample_window_indices", _fake_sample_window_indices)

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))
    grid = SamplingGrid(
        start_ns=10,
        exclusive_end_ns=11,
        timestamps_ns=np.array([10], dtype=np.int64),
        stride_ns=100,
        duration_ns=100,
    )

    sampled_batches = list(sensor.sample(SamplingSpec(grid=grid)))
    assert len(sampled_batches) == 1
    sampled = sampled_batches[0]

    assert sampled.align_timestamps_ns.shape == (0,)
    assert sampled.sensor_timestamps_ns.shape == (0,)
    assert sampled.frames.shape == (0, 3, 4, 3)


@pytest.mark.skip(
    reason=(
        "Last element in grid is exclusive-end marker; resolve_sensor_timestamps() "
        "treats it as active, needs to be changed."
    ),
)
def test_image_sensor_sample_preserves_one_output_row_per_align_timestamp(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling should delegate with ``dedup=False`` so rows stay aligned to active timestamps."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    def _fake_sample_window_indices(
        sensor_timestamps_ns: np.ndarray,
        window: SamplingWindow,
        *,
        policy: object,
        dedup: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        del sensor_timestamps_ns, policy
        assert dedup is False
        np.testing.assert_array_equal(window, np.array([10, 11], dtype=np.int64))
        return np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)

    monkeypatch.setattr(image_sensor_module, "sample_window_indices", _fake_sample_window_indices)

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10, 11], dtype=np.int64))
    grid = SamplingGrid(
        start_ns=10,
        exclusive_end_ns=11,
        timestamps_ns=np.array([10], dtype=np.int64),
        stride_ns=100,
        duration_ns=100,
    )

    sampled = next(sensor.sample(SamplingSpec(grid=grid)))

    np.testing.assert_array_equal(sampled.align_timestamps_ns, np.array([10], dtype=np.int64))
    np.testing.assert_array_equal(sampled.sensor_timestamps_ns, np.array([10], dtype=np.int64))
    assert sampled.frames.shape == (1, 3, 4, 3)
    assert tuple(sampled.frames[0, 0, 0]) == (255, 0, 0)


def test_image_sensor_sample_with_no_windows_returns_nothing(tmp_path: pathlib.Path) -> None:
    """If the grid yields no windows, sampling should yield no ImageData batches."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))

    assert list(sensor.sample(_StaticSpec([]))) == []


def test_image_sensor_empty_image_data_is_cached(tmp_path: pathlib.Path) -> None:
    """The cached empty ImageData instance should be reused."""
    image_a = tmp_path / "a.png"
    _write_image(image_a, (255, 0, 0))

    sensor = ImageSensor([image_a], sensor_timestamps_ns=np.array([10], dtype=np.int64))

    assert sensor._get_empty_image_data() is sensor._get_empty_image_data()


def test_image_sensor_path_and_stream_sources_produce_equivalent_output(
    tmp_path: pathlib.Path,
) -> None:
    """A sensor built from a Path and one built from a BytesIO of the same bytes match.

    Verifies the new file-like ``DataSource`` arm by feeding the same image
    via a :class:`pathlib.Path` and via a fresh :class:`io.BytesIO` and
    asserting the sampled output is byte-identical.
    """
    image_path = tmp_path / "img.png"
    _write_image(image_path, (123, 45, 200))
    raw_bytes = image_path.read_bytes()

    path_sensor = ImageSensor([image_path], sensor_timestamps_ns=np.array([10], dtype=np.int64))
    stream_sensor = ImageSensor([io.BytesIO(raw_bytes)], sensor_timestamps_ns=np.array([10], dtype=np.int64))

    grid = SamplingGrid(
        start_ns=10,
        exclusive_end_ns=11,
        timestamps_ns=np.array([10], dtype=np.int64),
        stride_ns=1,
        duration_ns=1,
    )

    path_sample = next(path_sensor.sample(SamplingSpec(grid=grid)))
    stream_sample = next(stream_sensor.sample(SamplingSpec(grid=grid)))

    np.testing.assert_array_equal(path_sample.frames, stream_sample.frames)
    np.testing.assert_array_equal(path_sample.sensor_timestamps_ns, stream_sample.sensor_timestamps_ns)
    np.testing.assert_array_equal(path_sample.align_timestamps_ns, stream_sample.align_timestamps_ns)
    assert path_sample.metadata == stream_sample.metadata
