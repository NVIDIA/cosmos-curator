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
"""Tests for CameraSensor.

Because CameraSensor is an orchestrator, it delegates most of its work.

To test, many of the dependencies, particularly the video decoding,
are mocked and the interactions checked.
"""

import io
from collections.abc import Callable
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.intrinsics import CameraIntrinsics
from cosmos_curator.core.sensors.data.video import VideoIndex, VideoMetadata
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import SamplingPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.utils.video import GpuVideoDecodeConfig, VideoDecodeConfig
from tests.cosmos_curator.core.sensors.test_utils import make_sampling_grid


def _make_video_index_and_metadata(  # noqa: PLR0913
    *,
    pts_ns: list[int],
    pts_stream: list[int],
    is_keyframe: list[bool],
    is_discard: list[bool],
    kf_pts_ns: list[int],
    kf_pts_stream: list[int],
    height: int = 2,
    width: int = 2,
) -> tuple[VideoIndex, VideoMetadata]:
    """Create a small fake indexed video stream for CameraSensor tests."""
    n_packets = len(pts_ns)
    index = VideoIndex(
        offset=np.arange(0, n_packets * 10, 10, dtype=np.int64),
        size=np.full(n_packets, 100, dtype=np.int64),
        pts_ns=np.array(pts_ns, dtype=np.int64),
        pts_stream=np.array(pts_stream, dtype=np.int64),
        is_keyframe=np.array(is_keyframe, dtype=np.bool_),
        is_discard=np.array(is_discard, dtype=np.bool_),
        kf_pts_ns=np.array(kf_pts_ns, dtype=np.int64),
        kf_pts_stream=np.array(kf_pts_stream, dtype=np.int64),
        time_base=Fraction(1, 1_000_000_000),
    )
    metadata = VideoMetadata(
        codec_name="h264",
        codec_max_bframes=0,
        codec_profile="",
        container_format="mp4",
        height=height,
        width=width,
        avg_frame_rate=Fraction(30, 1),
        pix_fmt="yuv420p",
        bit_rate_bps=1,
    )
    return index, metadata


class _FakeDecoder:
    """Minimal decoder context manager for CameraSensor test doubles."""

    def __init__(
        self,
        *,
        time_base: Fraction,
        decode_fn: Callable[[list[tuple[int, list[tuple[int, int]]]]], npt.NDArray[np.uint8]],
    ) -> None:
        self._time_base = time_base
        self._decode_fn = decode_fn

    def __enter__(self) -> SimpleNamespace:
        return SimpleNamespace(time_base=self._time_base, decode=self._decode_fn)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb


def _make_extrinsics() -> SensorExtrinsics:
    """Build a minimal SensorExtrinsics instance for CameraSensor tests."""
    return SensorExtrinsics(matrix=np.eye(4, dtype=np.float64))


def _make_intrinsics(*, width: int = 2, height: int = 2) -> CameraIntrinsics:
    """Build a minimal CameraIntrinsics instance for CameraSensor tests."""
    return CameraIntrinsics(
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coefficients=np.zeros(5, dtype=np.float64),
        distortion_model="brown_conrady",
        width=width,
        height=height,
    )


@pytest.fixture
def patch_camera_sensor_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., None]:
    """Patch the CameraSensor dependency seams used by these tests."""

    def _patch(
        *,
        make_index_and_metadata_fn: Callable[..., tuple[VideoIndex, VideoMetadata]],
        decoder_open_fn: Callable[..., _FakeDecoder] | None = None,
        sample_window_indices_fn: Callable[..., tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] | None = None,
    ) -> None:
        monkeypatch.setattr(
            "cosmos_curator.core.sensors.sensors.camera_sensor.make_index_and_metadata",
            make_index_and_metadata_fn,
        )
        if decoder_open_fn is not None:
            monkeypatch.setattr(
                "cosmos_curator.core.sensors.sensors.camera_sensor.CpuVideoDecoder.open",
                decoder_open_fn,
            )
        if sample_window_indices_fn is not None:
            monkeypatch.setattr(
                "cosmos_curator.core.sensors.sensors.camera_sensor.sample_window_indices",
                sample_window_indices_fn,
            )

    return _patch


@pytest.fixture
def synthetic_video() -> io.BytesIO:
    """10-frame H.264 video at 30 fps, encoded in memory."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=30)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    stream.codec_context.max_b_frames = 0  # disable B-frames for predictable PTS values
    for i in range(10):
        array = np.full((stream.height, stream.width, 3), i * 20, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    buf.seek(0)
    return buf


def test_sample_yields_empty_camera_data_for_empty_windows(synthetic_video: io.BytesIO) -> None:
    """CameraSensor.sample must yield an empty CameraData for windows with no reference timestamps.

    Regression test for C5: when SamplingGrid.__iter__ yields an empty array for a
    window (no reference timestamps fall in that time range), CameraSensor.sample must
    still yield a CameraData with zero frames rather than crashing or silently skipping
    the window.  Preserving one yield per window maintains the invariant that result
    index i corresponds to the window starting at start_ns + i * stride_ns, which is
    required for correct multi-sensor alignment in SensorSession.

    Setup
    -----
    Reference grid: [0ms, 33ms, 900ms, 933ms] — large gap between 33ms and 900ms.
    SamplingGrid: stride=400ms, duration=200ms.

    Windows produced by SamplingGrid.__iter__:
      window 0  [0ms,  200ms] → [0ms, 33ms]        non-empty → frames.shape[0] > 0
      window 1  [400ms, 600ms] → []                EMPTY     → frames.shape[0] == 0
      window 2  [800ms, 1000ms] → [900ms, 933ms]   non-empty → frames.shape[0] > 0

    Before fix: raises ValueError("grid must be non-empty") on window 1.
    After fix:  yields three CameraData objects; window 1 has zero frames.
    """
    sensor = CameraSensor(synthetic_video.getvalue())
    pts = sensor.video_index.pts_ns  # actual frame PTS values in nanoseconds

    # Build a reference grid from real PTS values with a deliberate gap in the middle.
    # Using only actual PTS values keeps every nearest-neighbour delta at exactly 0,
    # so tolerance_ns=0 (the default) is satisfied without special-casing.
    #
    # Layout for a 10-frame 30fps video (pts[i] ≈ i * 33ms):
    #   first cluster : pts[0..2]  ≈ [0, 33, 66ms]
    #   gap           : pts[3..6]  removed — nothing in [66ms, 233ms]
    #   second cluster: pts[7..9]  ≈ [233, 266, 300ms]
    #   sentinel      : pts[-1]+1  exclusive upper bound for sample_closest_indices
    sentinel = int(pts[-1]) + 1
    ref_timestamps = np.concatenate([pts[:3], pts[-3:], [sentinel]])

    # 100ms stride, 70ms duration:
    #   window at pts[0] =   0ms: covers [  0ms,  70ms] → [pts[0], pts[1], pts[2]] non-empty
    #   window at         100ms: covers [100ms, 170ms] → []                        EMPTY
    #   window at         200ms: covers [200ms, 270ms] → [pts[7], pts[8]]          non-empty
    #   window at         300ms: covers [300ms, 370ms] → [pts[9], sentinel]        non-empty
    stride_ns = 100_000_000  # 100ms
    duration_ns = 70_000_000  # 70ms — spans ~2 frames at 30fps

    grid = make_sampling_grid(timestamps_ns=ref_timestamps, stride_ns=stride_ns, duration_ns=duration_ns)
    spec = SamplingSpec(grid=grid)

    n_windows = sum(1 for _ in grid)
    results = list(sensor.sample(spec))

    # One result per window — index i always maps to start_ns + i * stride_ns.
    assert len(results) == n_windows

    frame_counts = [r.frames.shape[0] for r in results]

    # At least one window in the gap must have been empty.
    assert 0 in frame_counts

    # At least one window outside the gap must have decoded frames.
    assert any(n > 0 for n in frame_counts)


def test_sample_boundary_timestamp_belongs_to_next_window(synthetic_video: io.BytesIO) -> None:
    """A timestamp exactly on a window boundary must be emitted only by the later batch."""
    sensor = CameraSensor(synthetic_video.getvalue())
    pts = sensor.video_index.pts_ns

    # Two adjacent windows share pts[3] as the boundary marker:
    #   window 0 -> [pts[0], pts[1], pts[2], pts[3]]
    #   window 1 -> [pts[3], pts[4], pts[5], pts[6]]
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=pts[:7], stride_ns=int(pts[3] - pts[0]), duration_ns=int(pts[3] - pts[0])
        ),
    )

    batches = list(sensor.sample(spec))
    assert len(batches) == 2

    np.testing.assert_array_equal(batches[0].align_timestamps_ns, pts[:3])
    np.testing.assert_array_equal(batches[1].align_timestamps_ns, pts[3:6])
    np.testing.assert_array_equal(batches[0].sensor_timestamps_ns, pts[:3])
    np.testing.assert_array_equal(batches[1].sensor_timestamps_ns, pts[3:6])


def test_sample_singleton_window_is_boundary_only() -> None:
    """A one-frame source yields a singleton window and therefore an empty batch."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=30)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    stream.codec_context.max_b_frames = 0

    frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    buf.seek(0)

    sensor = CameraSensor(buf.getvalue())
    spec = SamplingSpec(grid=make_sampling_grid(sensor.timestamps_ns, stride_ns=1, duration_ns=1))

    batches = list(sensor.sample(spec))
    assert len(batches) == 1
    assert batches[0].align_timestamps_ns.shape == (0,)
    assert batches[0].sensor_timestamps_ns.shape == (0,)
    assert batches[0].pts_stream.shape == (0,)
    assert batches[0].frames.shape == (0, 16, 16, 3)


def test_camera_sensor_does_not_close_caller_owned_binaryio(synthetic_video: io.BytesIO) -> None:
    """CameraSensor should not close a caller-owned BinaryIO during init or sampling."""
    sensor = CameraSensor(synthetic_video)
    assert not synthetic_video.closed

    pts = sensor.timestamps_ns
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=pts[:4], stride_ns=int(pts[1] - pts[0]), duration_ns=int(pts[1] - pts[0])
        ),
    )

    batches = list(sensor.sample(spec))

    assert batches
    assert not synthetic_video.closed
    synthetic_video.seek(0)


def test_camera_sensor_rejects_unsupported_decode_config(synthetic_video: io.BytesIO) -> None:
    """CameraSensor should raise a clear error for unsupported decode config subclasses."""
    sensor = CameraSensor(synthetic_video.getvalue(), decode_config=VideoDecodeConfig())
    spec = SamplingSpec(grid=make_sampling_grid(sensor.timestamps_ns[:2], stride_ns=1, duration_ns=1))

    with pytest.raises(ValueError, match="unsupported decode_config"):
        next(sensor.sample(spec))


def test_camera_sensor_uses_only_displayable_frames(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """Discard packets should stay out of public sampled output."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[10, 20, 30],
        is_keyframe=[True, False, False],
        is_discard=[False, True, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )

    decode_calls: list[list[tuple[int, list[tuple[int, int]]]]] = []

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def _decode_with_capture(decode_plan: list[tuple[int, list[tuple[int, int]]]]) -> npt.NDArray[np.uint8]:
        decode_calls.append(decode_plan)
        total = sum(count for _, group in decode_plan for _, count in group)
        return np.zeros((total, metadata.height, metadata.width, 3), dtype=np.uint8)

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([0, 1], dtype=np.int64), np.array([2, 1], dtype=np.int64)

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(time_base=index.time_base, decode_fn=_decode_with_capture)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used")

    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([100, 200, 300, 301], dtype=np.int64),
            stride_ns=1_000,
            duration_ns=1_000,
        )
    )
    batches = list(sensor.sample(spec))

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0].align_timestamps_ns, np.array([100, 200, 300], dtype=np.int64))
    np.testing.assert_array_equal(batches[0].sensor_timestamps_ns, np.array([10, 10, 30], dtype=np.int64))
    np.testing.assert_array_equal(batches[0].pts_stream, np.array([10, 10, 30], dtype=np.int64))
    assert batches[0].frames.shape == (3, 2, 2, 3)
    assert decode_calls == [[(10, [(10, 2), (30, 1)])]]


def test_camera_sensor_rejects_stream_with_only_discard_packets(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """A sensor must expose at least one displayable frame."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100],
        pts_stream=[10],
        is_keyframe=[True],
        is_discard=[True],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
    )

    with pytest.raises(ValueError, match="no displayable frames"):
        CameraSensor(source=b"not-used")


def test_camera_sensor_passes_window_to_sample_window_indices(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should pass display timestamps and the full yielded window to the sampler."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[10, 20, 30],
        is_keyframe=[True, False, False],
        is_discard=[False, True, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    sampling_calls: list[tuple[npt.NDArray[np.int64], SamplingWindow]] = []

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        # Silence unused parameter warnings.
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        # Silence unused parameter warnings.
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros((3, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        window: SamplingWindow,
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del policy, dedup
        sampling_calls.append((canonical.copy(), window))
        return np.array([0, 1], dtype=np.int64), np.array([2, 1], dtype=np.int64)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 300, 301], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batches = list(sensor.sample(SamplingSpec(grid=grid)))

    assert len(batches) == 1
    assert len(sampling_calls) == 1
    np.testing.assert_array_equal(sampling_calls[0][0], np.array([100, 300], dtype=np.int64))
    np.testing.assert_array_equal(sampling_calls[0][1].timestamps_ns, np.array([100, 200, 300], dtype=np.int64))
    assert sampling_calls[0][1].start_ns == 100
    assert sampling_calls[0][1].exclusive_end_ns == 301


def test_camera_sensor_expands_repeated_picks_into_aligned_rows(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """Repeated sampler picks should expand back to one output row per reference timestamp."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[10, 20, 30],
        is_keyframe=[True, False, False],
        is_discard=[False, True, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    decode_calls: list[list[tuple[int, list[tuple[int, int]]]]] = []

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def _decode_with_capture(decode_plan: list[tuple[int, list[tuple[int, int]]]]) -> npt.NDArray[np.uint8]:
        decode_calls.append(decode_plan)
        total = sum(count for _, group in decode_plan for _, count in group)
        return np.zeros((total, metadata.height, metadata.width, 3), dtype=np.uint8)

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(time_base=index.time_base, decode_fn=_decode_with_capture)

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([0, 1], dtype=np.int64), np.array([2, 1], dtype=np.int64)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 300, 301], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100, 200, 300], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([10, 10, 30], dtype=np.int64))
    np.testing.assert_array_equal(batch.pts_stream, np.array([10, 10, 30], dtype=np.int64))
    assert batch.frames.shape == (3, 2, 2, 3)
    assert (
        len(batch.align_timestamps_ns)
        == len(batch.sensor_timestamps_ns)
        == len(batch.pts_stream)
        == batch.frames.shape[0]
    )
    assert decode_calls == [[(10, [(10, 2), (30, 1)])]]


def test_camera_sensor_returns_empty_when_window_has_no_displayable_matches(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """A non-empty reference window can still produce an empty sampled payload."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[10, 20, 30],
        is_keyframe=[True, False, False],
        is_discard=[False, False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    decode_calls: list[list[tuple[int, list[tuple[int, int]]]]] = []

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def _decode_with_capture(decode_plan: list[tuple[int, list[tuple[int, int]]]]) -> npt.NDArray[np.uint8]:
        decode_calls.append(decode_plan)
        return np.empty((0, metadata.height, metadata.width, 3), dtype=np.uint8)

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(time_base=index.time_base, decode_fn=_decode_with_capture)

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([150, 250, 350], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    assert batch.align_timestamps_ns.shape == (0,)
    assert batch.sensor_timestamps_ns.shape == (0,)
    assert batch.pts_stream.shape == (0,)
    assert batch.frames.shape == (0, 2, 2, 3)
    assert decode_calls == []


def test_camera_sensor_propagates_extrinsics_to_sampled_batches(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should attach configured extrinsics to decoded batches."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200],
        pts_stream=[10, 20],
        is_keyframe=[True, False],
        is_discard=[False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    extrinsics = _make_extrinsics()

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros((2, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([0, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used", extrinsics=extrinsics)
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 201], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    assert batch.extrinsics is extrinsics


def test_camera_sensor_defaults_extrinsics_to_none(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should keep extrinsics optional for existing callers."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200],
        pts_stream=[10, 20],
        is_keyframe=[True, False],
        is_discard=[False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros((2, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([0, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 201], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    assert batch.extrinsics is None


def test_camera_sensor_preserves_provided_intrinsics(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should attach configured intrinsics to sampled batches."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200],
        pts_stream=[10, 20],
        is_keyframe=[True, False],
        is_discard=[False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    intrinsics = _make_intrinsics(width=metadata.width, height=metadata.height)

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros((2, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([0, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used", intrinsics=intrinsics)
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 201], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    assert batch.intrinsics is intrinsics


def test_camera_sensor_defaults_intrinsics_to_none(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should keep intrinsics optional for existing callers."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200],
        pts_stream=[10, 20],
        is_keyframe=[True, False],
        is_discard=[False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros((2, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([0, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 201], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    assert batch.intrinsics is None


def test_camera_sensor_empty_batches_preserve_extrinsics(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should attach configured extrinsics to cached empty batches."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200],
        pts_stream=[10, 20],
        is_keyframe=[True, False],
        is_discard=[False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    extrinsics = _make_extrinsics()

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.empty((0, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used", extrinsics=extrinsics)
    grid = make_sampling_grid(
        timestamps_ns=np.array([150, 250], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    empty0 = next(sensor.sample(SamplingSpec(grid=grid)))
    empty1 = sensor._get_empty_camera_data()

    assert empty0.extrinsics is extrinsics
    assert empty1.extrinsics is extrinsics
    assert empty0 is empty1


def test_camera_sensor_empty_batches_preserve_intrinsics(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should attach configured intrinsics to cached empty batches."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200],
        pts_stream=[10, 20],
        is_keyframe=[True, False],
        is_discard=[False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    intrinsics = _make_intrinsics(width=metadata.width, height=metadata.height)

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.empty((0, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used", intrinsics=intrinsics)
    grid = make_sampling_grid(
        timestamps_ns=np.array([150, 250], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    empty0 = next(sensor.sample(SamplingSpec(grid=grid)))
    empty1 = sensor._get_empty_camera_data()

    assert empty0.intrinsics is intrinsics
    assert empty1.intrinsics is intrinsics
    assert empty0 is empty1


def test_camera_sensor_propagates_sampling_policy_failures(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """Sampling policy failures should propagate unchanged through CameraSensor.sample()."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[10, 20, 30],
        is_keyframe=[True, False, False],
        is_discard=[False, False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros((0, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, dedup
        assert isinstance(policy, SamplingPolicy)
        msg = "tolerance_ns=5 exceeded: max delta was 10 ns for grid=200, canonical=190"
        raise ValueError(msg)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used")
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([100, 200, 300], dtype=np.int64),
            stride_ns=1_000,
            duration_ns=1_000,
        ),
        policy=SamplingPolicy(tolerance_ns=5),
    )

    with pytest.raises(ValueError, match="tolerance_ns=5 exceeded"):
        next(sensor.sample(spec))


def test_camera_sensor_uses_display_pts_stream_sidecar_alignment(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """Discarded packet stream timestamps must not appear in sampled sidecar outputs."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[10, 20, 30],
        is_keyframe=[True, False, False],
        is_discard=[False, True, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros((2, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    def fake_sample_window_indices(
        canonical: npt.NDArray[np.int64],
        grid: npt.NDArray[np.int64],
        *,
        policy: object = None,
        dedup: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del canonical, grid, policy, dedup
        return np.array([0, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
        sample_window_indices_fn=fake_sample_window_indices,
    )

    sensor = CameraSensor(b"not-used")
    grid = make_sampling_grid(
        timestamps_ns=np.array([100, 200, 300], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batch = next(sensor.sample(SamplingSpec(grid=grid)))

    np.testing.assert_array_equal(batch.pts_stream, np.array([10, 30], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([10, 30], dtype=np.int64))
    assert 20 not in batch.pts_stream.tolist()


def test_camera_sensor_public_properties(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should expose the indexed display timeline and metadata through its public properties."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[10, 20, 30],
        is_keyframe=[True, False, False],
        is_discard=[False, True, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    patch_camera_sensor_dependencies(make_index_and_metadata_fn=fake_make_index_and_metadata)

    sensor = CameraSensor(b"not-used")

    assert sensor.video_index is index
    assert sensor.video_metadata is metadata
    assert sensor.start_ns == 100
    assert sensor.end_ns == 300
    assert sensor.max_gap_ns == 0
    np.testing.assert_array_equal(sensor.timestamps_ns, np.array([100, 300], dtype=np.int64))
    assert sensor.codec_name == "h264"
    assert sensor.codec_max_bframes == 0


def test_camera_sensor_sample_supports_gpu_decode_config(
    patch_camera_sensor_dependencies: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CameraSensor.sample should use the GPU decoder branch when configured with GpuVideoDecodeConfig."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[10, 20, 30],
        is_keyframe=[True, False, False],
        is_discard=[False, False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    gpu_open_calls: list[tuple[object, int, object, object]] = []

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_gpu_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
    ) -> _FakeDecoder:
        gpu_open_calls.append((source, stream_idx, config, stats))
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros((2, metadata.height, metadata.width, 3), dtype=np.uint8),  # noqa: ARG005
        )

    patch_camera_sensor_dependencies(make_index_and_metadata_fn=fake_make_index_and_metadata)
    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.camera_sensor.GpuVideoDecoder.open",
        fake_gpu_decoder_open,
    )

    decode_config = GpuVideoDecodeConfig()
    sensor = CameraSensor(b"not-used", decode_config=decode_config)
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([100, 200, 300], dtype=np.int64),
            stride_ns=1_000,
            duration_ns=1_000,
        )
    )

    batch = next(sensor.sample(spec))

    assert len(gpu_open_calls) == 1
    assert gpu_open_calls[0][0] == b"not-used"
    assert gpu_open_calls[0][1] == 0
    assert gpu_open_calls[0][2] is decode_config
    assert gpu_open_calls[0][3] is None
    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
