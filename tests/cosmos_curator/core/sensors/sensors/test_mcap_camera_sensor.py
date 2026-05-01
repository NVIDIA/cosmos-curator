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
"""Tests for McapCameraSensor."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.video import VideoMetadata
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.mcap_camera_sensor import (
    McapCameraSensor,
    _rgb8_channel_dimensions,
)
from tests.cosmos_curator.core.sensors.test_utils import make_sampling_grid


def _make_metadata(*, width: int = 2, height: int = 2) -> VideoMetadata:
    """Build minimal VideoMetadata for McapCameraSensor tests."""
    return VideoMetadata(
        codec_name="rgb8",
        codec_max_bframes=0,
        codec_profile="",
        container_format="mcap",
        height=height,
        width=width,
        avg_frame_rate=Fraction(30, 1),
        pix_fmt="rgb24",
        bit_rate_bps=1,
    )


class _FakeReader:
    """Minimal fake MCAP reader."""

    def __init__(self, *, summary: object = None) -> None:
        self._summary = summary

    def get_summary(self) -> object:
        return self._summary


@pytest.fixture
def patch_mcap_camera_sensor_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., None]:
    """Patch the McapCameraSensor dependency seams used by these tests."""

    def _patch(**patches: Callable[..., Any]) -> None:
        for name, value in patches.items():
            monkeypatch.setattr(f"cosmos_curator.core.sensors.sensors.mcap_camera_sensor.{name}", value)

    return _patch


def test_mcap_camera_sensor_timestamps_ns_caches_timeline(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """timestamps_ns should load the timeline once and reuse it for start/end."""
    timeline_calls = 0
    start_end_calls = 0
    timeline = np.array([100, 200, 300], dtype=np.int64)

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader()

    def fake_load_timeline(reader: object, topic: str) -> npt.NDArray[np.int64]:
        nonlocal timeline_calls
        del reader, topic
        timeline_calls += 1
        return timeline

    def fake_load_start_end_ns(reader: object, topic: str) -> tuple[int, int]:
        nonlocal start_end_calls
        del reader, topic
        start_end_calls += 1
        return 0, 0

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        load_timeline=fake_load_timeline,
        load_start_end_ns=fake_load_start_end_ns,
    )

    sensor = McapCameraSensor(b"not-used")

    np.testing.assert_array_equal(sensor.timestamps_ns, timeline)
    assert sensor.start_ns == 100
    assert sensor.end_ns == 300
    assert timeline_calls == 1
    assert start_end_calls == 0


def test_mcap_camera_sensor_start_end_ns_can_load_without_full_timeline(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """start_ns/end_ns should not require loading the full timeline."""
    timeline_calls = 0
    start_end_calls = 0

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader()

    def fake_load_timeline(reader: object, topic: str) -> npt.NDArray[np.int64]:
        nonlocal timeline_calls
        del reader, topic
        timeline_calls += 1
        return np.array([100, 200, 300], dtype=np.int64)

    def fake_load_start_end_ns(reader: object, topic: str) -> tuple[int, int]:
        nonlocal start_end_calls
        del reader, topic
        start_end_calls += 1
        return 100, 300

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        load_timeline=fake_load_timeline,
        load_start_end_ns=fake_load_start_end_ns,
    )

    sensor = McapCameraSensor(b"not-used")

    assert sensor.start_ns == 100
    assert sensor.end_ns == 300
    assert start_end_calls == 1
    assert timeline_calls == 0


def test_mcap_camera_sensor_samples_window_and_reports_nanosecond_pts_stream(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """McapCameraSensor should sample one window and report pts_stream in nanoseconds."""
    metadata = _make_metadata(width=2, height=2)
    frame0 = bytes(range(12))
    frame1 = bytes(range(12, 24))
    sampling_calls: list[tuple[npt.NDArray[np.int64], SamplingWindow]] = []

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=None)

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    def fake_iter_messages_log_time_ns(
        reader: object,
        topic: str,
        start_ns: int,
        end_ns_exclusive: int,
        *,
        log_time_order: bool = True,
    ) -> Iterator[tuple[object, object, object]]:
        del reader, topic, start_ns, end_ns_exclusive, log_time_order
        yield None, None, SimpleNamespace(log_time=100, data=frame0)
        yield None, None, SimpleNamespace(log_time=300, data=frame1)

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        window: SamplingWindow,
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del policy
        assert dedup is False
        sampling_calls.append((canonical.copy(), window))
        return np.array([0, 0, 1], dtype=np.int64), np.array([1, 1, 1], dtype=np.int64)

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
        iter_messages_log_time_ns=fake_iter_messages_log_time_ns,
        sample_window_indices=fake_sample_window_indices,
    )

    sensor = McapCameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 300, 301], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    assert len(sampling_calls) == 1
    np.testing.assert_array_equal(sampling_calls[0][0], np.array([100, 300], dtype=np.int64))
    np.testing.assert_array_equal(sampling_calls[0][1].timestamps_ns, np.array([100, 200, 300], dtype=np.int64))
    assert sampling_calls[0][1].start_ns == 100
    assert sampling_calls[0][1].exclusive_end_ns == 301
    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100, 200, 300], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100, 100, 300], dtype=np.int64))
    np.testing.assert_array_equal(batch.pts_stream, np.array([100, 100, 300], dtype=np.int64))
    assert batch.frames.shape == (3, 2, 2, 3)


def test_mcap_camera_sensor_returns_empty_batch_when_window_has_no_messages(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """A window with no topic messages should yield an empty CameraData batch."""
    metadata = _make_metadata(width=2, height=2)

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=None)

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    def fake_iter_messages_log_time_ns(
        reader: object,
        topic: str,
        start_ns: int,
        end_ns_exclusive: int,
        *,
        log_time_order: bool = True,
    ) -> Iterator[tuple[object, object, object]]:
        del reader, topic, start_ns, end_ns_exclusive, log_time_order
        if False:
            yield None, None, None
        return

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
        iter_messages_log_time_ns=fake_iter_messages_log_time_ns,
    )

    sensor = McapCameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 300], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    assert batch.align_timestamps_ns.shape == (0,)
    assert batch.sensor_timestamps_ns.shape == (0,)
    assert batch.pts_stream.shape == (0,)
    assert batch.frames.shape == (0, 2, 2, 3)


def test_mcap_camera_sensor_rejects_bad_rgb8_payload_size(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """MCAP payloads must match width*height*3 bytes for rgb8 frames."""
    metadata = _make_metadata(width=2, height=2)

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=None)

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    def fake_iter_messages_log_time_ns(
        reader: object,
        topic: str,
        start_ns: int,
        end_ns_exclusive: int,
        *,
        log_time_order: bool = True,
    ) -> Iterator[tuple[object, object, object]]:
        del reader, topic, start_ns, end_ns_exclusive, log_time_order
        yield None, None, SimpleNamespace(log_time=100, data=b"too-short")

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
        iter_messages_log_time_ns=fake_iter_messages_log_time_ns,
    )

    sensor = McapCameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    with pytest.raises(ValueError, match=r"rgb8 payload size"):
        next(sensor.sample(SamplingSpec(grid=grid)))


def test_mcap_camera_sensor_rejects_summary_dimensions_that_disagree_with_metadata(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """Summary channel dimensions should agree with the stored metadata record."""
    metadata = _make_metadata(width=2, height=2)
    fake_channel = SimpleNamespace(
        topic="/camera/rgb",
        message_encoding="rgb8",
        metadata={"width": "3", "height": "2"},
    )

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=SimpleNamespace(channels={1: fake_channel}))

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    def fake_channel_for_topic(summary: object, topic: str) -> object:
        del summary, topic
        return fake_channel

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
        channel_for_topic=fake_channel_for_topic,
    )

    sensor = McapCameraSensor(b"not-used")

    with pytest.raises(ValueError, match=r"MCAP channel dimensions do not match stored video metadata"):
        next(sensor.sample(SamplingSpec(grid=make_sampling_grid(np.array([100, 200], dtype=np.int64), 1_000, 1_000))))


@pytest.mark.parametrize(
    ("channel", "raises", "expected"),
    [
        (
            SimpleNamespace(topic="/camera/rgb", message_encoding="jpeg", metadata={"width": "2", "height": "2"}),
            pytest.raises(ValueError, match="expected rgb8 channel"),
            None,
        ),
        (
            SimpleNamespace(topic="/camera/rgb", message_encoding="rgb8", metadata={"width": "2"}),
            pytest.raises(ValueError, match="channel metadata must include integer width and height strings"),
            None,
        ),
        (
            SimpleNamespace(topic="/camera/rgb", message_encoding="rgb8", metadata={"width": "nope", "height": "2"}),
            pytest.raises(ValueError, match="channel metadata must include integer width and height strings"),
            None,
        ),
        (
            SimpleNamespace(topic="/camera/rgb", message_encoding="rgb8", metadata={"width": "0", "height": "2"}),
            pytest.raises(ValueError, match="invalid rgb8 dimensions"),
            None,
        ),
        (
            SimpleNamespace(topic="/camera/rgb", message_encoding="rgb8", metadata={"width": "2", "height": "3"}),
            nullcontext(),
            (2, 3),
        ),
    ],
)
def test_rgb8_channel_dimensions_contract(
    channel: object,
    raises: AbstractContextManager[object],
    expected: tuple[int, int] | None,
) -> None:
    """_rgb8_channel_dimensions should validate encoding and dimensions precisely."""
    with raises:
        dims = _rgb8_channel_dimensions(channel)  # type: ignore[arg-type]
        if expected is not None:
            assert dims == expected


def test_mcap_camera_sensor_cached_properties_and_timeline_cache_reuse() -> None:
    """Cached timeline data should drive timestamps_ns, start_ns, end_ns, and max_gap_ns."""
    sensor = McapCameraSensor(b"not-used")
    timeline = np.array([100, 200, 300], dtype=np.int64)
    sensor._message_log_times_ns_cache = timeline

    np.testing.assert_array_equal(sensor.timestamps_ns, timeline)
    np.testing.assert_array_equal(sensor.timestamps_ns, timeline)
    assert sensor.start_ns == 100
    assert sensor.end_ns == 300
    assert sensor.max_gap_ns == 0


def test_mcap_camera_sensor_start_end_properties_raise_if_cache_population_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start_ns/end_ns should fail clearly if cache population leaves the values unset."""
    sensor = McapCameraSensor(b"not-used")

    def fake_ensure_start_end_ns_cached() -> None:
        sensor._start_ns = None
        sensor._end_ns = None

    monkeypatch.setattr(sensor, "_ensure_start_end_ns_cached", fake_ensure_start_end_ns_cached)

    with pytest.raises(ValueError, match="start_ns was not loaded"):
        _ = sensor.start_ns

    with pytest.raises(ValueError, match="end_ns was not loaded"):
        _ = sensor.end_ns


def test_mcap_camera_sensor_resolve_topic_dimensions_uses_metadata_when_topic_is_missing(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """Missing topic summary info should fall back to the stored metadata dimensions."""
    metadata = _make_metadata(width=5, height=7)

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=SimpleNamespace(channels={}))

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    def fake_channel_for_topic(summary: object, topic: str) -> None:
        del summary, topic

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
        channel_for_topic=fake_channel_for_topic,
    )

    sensor = McapCameraSensor(b"not-used")
    width, height = sensor._resolve_topic_dimensions(_FakeReader(summary=SimpleNamespace(channels={})))

    assert width == 5
    assert height == 7


def test_mcap_camera_sensor_resolve_topic_dimensions_accepts_matching_channel_dimensions(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """Matching rgb8 channel dimensions should be accepted and returned."""
    metadata = _make_metadata(width=2, height=3)
    fake_channel = SimpleNamespace(
        topic="/camera/rgb",
        message_encoding="rgb8",
        metadata={"width": "2", "height": "3"},
    )

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=SimpleNamespace(channels={1: fake_channel}))

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    def fake_channel_for_topic(summary: object, topic: str) -> object:
        del summary, topic
        return fake_channel

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
        channel_for_topic=fake_channel_for_topic,
    )

    sensor = McapCameraSensor(b"not-used")
    width, height = sensor._resolve_topic_dimensions(_FakeReader(summary=SimpleNamespace(channels={1: fake_channel})))

    assert width == 2
    assert height == 3


def test_mcap_camera_sensor_read_window_messages_returns_empty_for_empty_window() -> None:
    """_read_window_messages should return empty outputs immediately for an empty window."""
    sensor = McapCameraSensor(b"not-used")

    log_times_ns, payloads = sensor._read_window_messages(
        _FakeReader(),
        SamplingWindow(start_ns=0, exclusive_end_ns=0, timestamps_ns=np.array([], dtype=np.int64)),
        width=2,
        height=2,
    )

    np.testing.assert_array_equal(log_times_ns, np.array([], dtype=np.int64))
    assert payloads == []


def test_mcap_camera_sensor_get_empty_camera_data_is_cached(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """_get_empty_camera_data should cache and reuse the empty CameraData batch."""
    metadata = _make_metadata(width=2, height=2)

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=None)

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
    )

    sensor = McapCameraSensor(b"not-used")
    empty0 = sensor._get_empty_camera_data()
    empty1 = sensor._get_empty_camera_data()

    assert empty0 is empty1
    assert empty0.frames.shape == (0, 2, 2, 3)


def test_mcap_camera_sensor_sample_window_returns_empty_when_sampler_selects_no_indices(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty message window can still produce an empty sampled batch."""
    metadata = _make_metadata(width=2, height=2)

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=None)

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
    )
    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.mcap_camera_sensor.sample_window_indices",
        fake_sample_window_indices,
    )

    sensor = McapCameraSensor(b"not-used")
    batch = sensor._sample_window(
        SamplingWindow(start_ns=100, exclusive_end_ns=200, timestamps_ns=np.array([100], dtype=np.int64)),
        np.array([100], dtype=np.int64),
        [bytes(range(12))],
        width=2,
        height=2,
        spec=SamplingSpec(grid=make_sampling_grid(np.array([100, 200], dtype=np.int64), 1_000, 1_000)),
    )

    assert batch.align_timestamps_ns.shape == (0,)
    assert batch.frames.shape == (0, 2, 2, 3)


def test_mcap_camera_sensor_sample_yields_empty_batch_for_empty_window(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """sample() should preserve empty windows and yield the cached empty batch for them."""
    metadata = _make_metadata(width=2, height=2)

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=None)

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
    )

    sensor = McapCameraSensor(b"not-used")
    spec = SimpleNamespace(grid=[np.array([], dtype=np.int64)])

    batch = next(sensor.sample(spec))  # type: ignore[arg-type]

    assert batch.align_timestamps_ns.shape == (0,)
    assert batch.sensor_timestamps_ns.shape == (0,)
    assert batch.pts_stream.shape == (0,)
    assert batch.frames.shape == (0, 2, 2, 3)


def test_mcap_camera_sensor_sample_returns_no_batches_when_grid_yields_nothing(
    patch_mcap_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """sample() should cleanly return when the provided grid yields no windows."""
    metadata = _make_metadata(width=2, height=2)

    @contextmanager
    def fake_open_data_source(source: object, mode: str = "rb") -> Iterator[object]:
        del source, mode
        yield object()

    def fake_make_reader(stream: object) -> _FakeReader:
        del stream
        return _FakeReader(summary=None)

    def fake_get_metadata_record(reader: object, name: str) -> dict[str, str]:
        del reader, name
        return metadata.to_string_dict()

    patch_mcap_camera_sensor_dependencies(
        open_data_source=fake_open_data_source,
        mcap_make_reader=fake_make_reader,
        get_metadata_record=fake_get_metadata_record,
    )

    sensor = McapCameraSensor(b"not-used")
    spec = SimpleNamespace(grid=[])

    assert list(sensor.sample(spec)) == []  # type: ignore[arg-type]
