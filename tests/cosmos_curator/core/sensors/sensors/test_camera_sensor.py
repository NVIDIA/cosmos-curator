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

from cosmos_curator.core.sensors.data.camera_data import MotionVectorData, MotionVectorFrameData
from cosmos_curator.core.sensors.data.extrinsics import SensorExtrinsics
from cosmos_curator.core.sensors.data.intrinsics import CameraIntrinsics
from cosmos_curator.core.sensors.data.video import VideoIndex, VideoMetadata
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.grid import SamplingWindow
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.core.sensors.utils.video import GpuVideoDecodeConfig, VideoDecodeConfig
from tests.cosmos_curator.core.sensors.test_utils import make_sampling_grid


def _make_video_index_and_metadata(
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
        has_bframes=False,
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
        decode_fn: Callable[
            [list[tuple[int, list[tuple[int, int]]]]],
            npt.NDArray[np.uint8] | tuple[npt.NDArray[np.uint8], MotionVectorData | None],
        ],
    ) -> None:
        self._time_base = time_base
        self._decode_fn = decode_fn

    def __enter__(self) -> SimpleNamespace:
        def _decode(
            decode_plan: list[tuple[int, list[tuple[int, int]]]],
        ) -> tuple[npt.NDArray[np.uint8], MotionVectorData | None]:
            result = self._decode_fn(decode_plan)
            if isinstance(result, tuple):
                return result
            return result, None

        return SimpleNamespace(time_base=self._time_base, decode=_decode)

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


def _make_motion_vector_data(count: int) -> MotionVectorData:
    """Build a minimal aligned motion-vector batch for CameraSensor tests."""
    return MotionVectorData(frames=tuple(MotionVectorFrameData.empty() for _ in range(count)))


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
    # so max_delta_ns=0 is satisfied without special-casing.
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
    results = list(sensor.sample(spec, policy=NearestTimestampPolicy()))

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

    batches = list(sensor.sample(spec, policy=NearestTimestampPolicy()))
    assert len(batches) == 2

    np.testing.assert_array_equal(batches[0].align_timestamps_ns, pts[:3])
    np.testing.assert_array_equal(batches[1].align_timestamps_ns, pts[3:6])
    np.testing.assert_array_equal(batches[0].sensor_timestamps_ns, pts[:3])
    np.testing.assert_array_equal(batches[1].sensor_timestamps_ns, pts[3:6])


@pytest.mark.parametrize("origin_ns", [1_000, -1_000, np.int32(1_000), np.uint64(1_000)])
def test_camera_sensor_origin_pins_the_first_frame_and_carries_the_rest_with_it(
    origin_ns: int,
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """``origin_ns`` states where the first frame lands, not how far to shift it.

    This index starts at 100 ns rather than zero, which is what separates the two
    readings: adding ``origin_ns`` would report the first frame at
    ``origin_ns + 100``. A caller holding a sensor timeline knows when recording
    started, not what the container chose to call that instant, so the sensor
    subtracts its own first PTS. Stream-native PTS are untouched, because decode
    planning and seeking still speak the container's timeline.
    """
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200, 300],
        pts_stream=[100, 200, 300],
        is_keyframe=[True, False, False],
        is_discard=[False, False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[100],
    )

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decode(decode_plan: list[tuple[int, list[tuple[int, int]]]]) -> npt.NDArray[np.uint8]:
        frame_count = sum(count for _, group in decode_plan for _, count in group)
        return np.zeros((frame_count, metadata.height, metadata.width, 3), dtype=np.uint8)

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(time_base=index.time_base, decode_fn=fake_decode)

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
    )

    normalized_origin_ns = int(origin_ns)
    sensor = CameraSensor(b"not-used", origin_ns=origin_ns)
    shifted_pts_ns = np.array([0, 100, 200], dtype=np.int64) + normalized_origin_ns
    grid_timestamps_ns = np.append(shifted_pts_ns, shifted_pts_ns[-1] + 1)
    grid = make_sampling_grid(grid_timestamps_ns, stride_ns=1_000, duration_ns=1_000)

    batch = next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy(max_delta_ns=0)))

    assert sensor.start_ns == normalized_origin_ns
    # The offset the origin resolved to, which is the origin less the container's
    # own first PTS of 100 -- not the origin itself.
    assert sensor.timestamp_offset_ns == normalized_origin_ns - 100
    assert type(sensor.timestamp_offset_ns) is int
    np.testing.assert_array_equal(sensor.video_index.pts_ns, shifted_pts_ns)
    np.testing.assert_array_equal(sensor.video_index.kf_pts_ns, shifted_pts_ns[:1])
    np.testing.assert_array_equal(sensor.timestamps_ns, shifted_pts_ns)
    assert sensor.start_ns == int(shifted_pts_ns[0])
    assert sensor.end_ns == int(shifted_pts_ns[-1])
    np.testing.assert_array_equal(next(sensor.stream_timestamps()), shifted_pts_ns)
    np.testing.assert_array_equal(batch.align_timestamps_ns, shifted_pts_ns)
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, shifted_pts_ns)
    np.testing.assert_array_equal(batch.pts_stream, np.array([100, 200, 300], dtype=np.int64))

    assert index.timestamp_offset_ns == 0
    np.testing.assert_array_equal(index.pts_ns, np.array([100, 200, 300], dtype=np.int64))


@pytest.mark.parametrize(
    "origin_ns",
    [
        True,
        np.bool_(True),  # noqa: FBT003
        1.5,
        "1",
        np.iinfo(np.int64).max + 1,
        np.uint64(np.iinfo(np.int64).max + 1),
    ],
)
def test_camera_sensor_rejects_an_invalid_origin_before_indexing(
    origin_ns: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable origin must fail before the video source is indexed."""

    def fail_to_index(*_args: object, **_kwargs: object) -> tuple[VideoIndex, VideoMetadata]:
        pytest.fail("CameraSensor must validate origin_ns before indexing")

    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.camera_sensor.make_index_and_metadata",
        fail_to_index,
    )

    with pytest.raises(ValueError, match="origin_ns"):
        CameraSensor(b"not-used", origin_ns=origin_ns)  # type: ignore[arg-type]


def _sensor_timeline(count: int) -> npt.NDArray[np.int64]:
    """Build a rig's own sensor times: epoch nanoseconds near 30 Hz, never exactly on it.

    Deliberately unlike anything a container would report -- epoch-scale rather than
    starting at zero, and unevenly spaced -- so a test cannot pass by accident on a
    timeline the index could also have produced.
    """
    base = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z
    jitter = np.array([0, 41_000, -37_000, 12_000, -8_000, 55_000, -19_000, 3_000, -44_000, 27_000], dtype=np.int64)
    return base + np.arange(count, dtype=np.int64) * 33_333_333 + jitter[:count]


def test_a_sensor_timeline_replaces_the_container_timeline(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """Every timestamp the sensor reports comes from the sensor timeline, and the index keeps its own.

    The split the feature rests on: what the sensor *reports* moves to the recording's
    clock, while the stream-native PTS the index holds stay exactly as they were,
    because those address frames within the file and are what seeking and decode
    planning speak.
    """
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[0, 33_333_333, 66_666_666],
        pts_stream=[0, 3_000, 6_000],
        is_keyframe=[True, False, False],
        is_discard=[False, False, False],
        kf_pts_ns=[0],
        kf_pts_stream=[0],
    )

    def fake_make_index_and_metadata(*_args: object, **_kwargs: object) -> tuple[VideoIndex, VideoMetadata]:
        return index, metadata

    patch_camera_sensor_dependencies(make_index_and_metadata_fn=fake_make_index_and_metadata)

    sensor_ns = _sensor_timeline(3)
    sensor = CameraSensor(b"not-used", sensor_timestamps_ns=sensor_ns)

    assert sensor.timestamp_source is TimestampSource.SENSOR
    np.testing.assert_array_equal(sensor.timestamps_ns, sensor_ns)
    np.testing.assert_array_equal(next(sensor.stream_timestamps()), sensor_ns)
    np.testing.assert_array_equal(np.concatenate(list(sensor.stream_timestamps(batch_size=2))), sensor_ns)
    assert sensor.start_ns == int(sensor_ns[0])
    assert sensor.end_ns == int(sensor_ns[-1])
    # Untouched, and no offset invented to reconcile the two clocks.
    np.testing.assert_array_equal(sensor.video_index.pts_ns, np.array([0, 33_333_333, 66_666_666], dtype=np.int64))
    np.testing.assert_array_equal(sensor.video_index.pts_stream, np.array([0, 3_000, 6_000], dtype=np.int64))
    assert sensor.timestamp_offset_ns == 0


def test_the_container_timeline_is_reported_when_no_sensor_timeline_is_given(
    synthetic_video: io.BytesIO,
) -> None:
    """The default is unchanged, and says so.

    This 30 fps fixture is the situation the feature exists for: a constant-frame-rate
    mux, whose PTS the encoder regenerated on a fixed grid. The sensor still reports it
    -- there is nothing else to report -- but labels it, which is what lets a reader
    tell a clean measurement from an unmeasurable one.
    """
    sensor = CameraSensor(synthetic_video.getvalue())

    assert sensor.timestamp_source is TimestampSource.CONTAINER_PTS
    np.testing.assert_array_equal(sensor.timestamps_ns, sensor.video_index.display_pts_ns)


def test_a_sensor_timeline_reaches_the_sampled_payload(synthetic_video: io.BytesIO) -> None:
    """A sampled batch is timestamped on the same clock the sensor reports.

    Runs through real decode rather than a fake one, because the payload's timestamps
    are the one place the two clocks could disagree without any property of the sensor
    itself looking wrong: a batch labelled in container PTS while ``start_ns`` reads
    epoch nanoseconds is not something a caller aligning two sensors could detect.
    """
    frame_count = len(CameraSensor(synthetic_video.getvalue()).timestamps_ns)
    sensor_ns = _sensor_timeline(frame_count)

    sensor = CameraSensor(synthetic_video.getvalue(), sensor_timestamps_ns=sensor_ns)
    stride_ns = int(sensor_ns[2] - sensor_ns[0])
    spec = SamplingSpec(grid=make_sampling_grid(sensor_ns[:5], stride_ns=stride_ns, duration_ns=stride_ns))

    batches = list(sensor.sample(spec, policy=NearestTimestampPolicy()))

    served = np.concatenate([batch.sensor_timestamps_ns for batch in batches])
    assert served.size
    # Every served timestamp is one the rig recorded, not one the container invented.
    assert set(served.tolist()) <= set(sensor_ns.tolist())
    np.testing.assert_array_equal(served, np.concatenate([batch.align_timestamps_ns for batch in batches]))
    # The frames themselves are still addressed by container PTS.
    pts_served = np.concatenate([batch.pts_stream for batch in batches])
    assert set(pts_served.tolist()) <= set(sensor.video_index.display_pts_stream.tolist())


def test_a_sensor_timeline_of_the_wrong_length_is_rejected(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """A timeline that does not describe this recording is an error, not something to trim.

    The two are matched by position, so a count that disagrees leaves every frame's
    sensor time in doubt rather than just the missing ones'.
    """
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[0, 100, 200],
        pts_stream=[0, 100, 200],
        is_keyframe=[True, False, False],
        is_discard=[False, False, False],
        kf_pts_ns=[0],
        kf_pts_stream=[0],
    )

    def fake_make_index_and_metadata(*_args: object, **_kwargs: object) -> tuple[VideoIndex, VideoMetadata]:
        return index, metadata

    patch_camera_sensor_dependencies(make_index_and_metadata_fn=fake_make_index_and_metadata)

    with pytest.raises(ValueError, match="2 observations but the stream has 3 displayable frames"):
        CameraSensor(b"not-used", sensor_timestamps_ns=_sensor_timeline(2))


def test_a_sensor_timeline_and_an_origin_cannot_be_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two answers to the same question are refused before the source is indexed."""

    def fail_to_index(*_args: object, **_kwargs: object) -> tuple[VideoIndex, VideoMetadata]:
        pytest.fail("CameraSensor must reject the combination before indexing")

    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.camera_sensor.make_index_and_metadata",
        fail_to_index,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        CameraSensor(b"not-used", origin_ns=1_000, sensor_timestamps_ns=_sensor_timeline(3))


@pytest.mark.parametrize(
    ("sensor_timestamps_ns", "match"),
    [
        (np.array([], dtype=np.int64), "at least one observation"),
        (np.array([100, 100], dtype=np.int64), "strictly sorted"),
        (np.array([200, 100], dtype=np.int64), "strictly sorted"),
        (np.array([[100, 200]], dtype=np.int64), "must be 1-D"),
        (np.array([100.5, 200.5], dtype=np.float64), "dtype int64"),
    ],
)
def test_an_unusable_sensor_timeline_is_rejected_before_indexing(
    sensor_timestamps_ns: npt.NDArray[np.int64],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeline no recording could have produced fails before the source is read.

    Checked up front for the reason an unusable origin is: the index is the expensive
    part, and none of these depend on it.
    """

    def fail_to_index(*_args: object, **_kwargs: object) -> tuple[VideoIndex, VideoMetadata]:
        pytest.fail("CameraSensor must validate sensor_timestamps_ns before indexing")

    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.camera_sensor.make_index_and_metadata",
        fail_to_index,
    )

    with pytest.raises(ValueError, match=match):
        CameraSensor(b"not-used", sensor_timestamps_ns=sensor_timestamps_ns)


def test_a_reported_sensor_timeline_cannot_be_written_through(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """The array the sensor hands back is a read-only view, as the index's own timeline is.

    Without this a caller could rewrite the sensor's clock in place after construction,
    past the validation that made it a usable timeline.
    """
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[0, 100, 200],
        pts_stream=[0, 100, 200],
        is_keyframe=[True, False, False],
        is_discard=[False, False, False],
        kf_pts_ns=[0],
        kf_pts_stream=[0],
    )

    def fake_make_index_and_metadata(*_args: object, **_kwargs: object) -> tuple[VideoIndex, VideoMetadata]:
        return index, metadata

    patch_camera_sensor_dependencies(make_index_and_metadata_fn=fake_make_index_and_metadata)

    sensor_ns = _sensor_timeline(3)
    sensor = CameraSensor(b"not-used", sensor_timestamps_ns=sensor_ns)

    with pytest.raises(ValueError, match="read-only"):
        sensor.timestamps_ns[0] = 0
    # The caller's own array stays writable: taking a view must not confiscate it.
    sensor_ns[0] = 0


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

    batches = list(sensor.sample(spec, policy=NearestTimestampPolicy()))
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

    batches = list(sensor.sample(spec, policy=NearestTimestampPolicy()))

    assert batches
    assert not synthetic_video.closed
    synthetic_video.seek(0)


def test_camera_sensor_rejects_unsupported_decode_config(synthetic_video: io.BytesIO) -> None:
    """CameraSensor should raise a clear error for unsupported decode config subclasses."""
    sensor = CameraSensor(synthetic_video.getvalue(), decode_config=VideoDecodeConfig())
    spec = SamplingSpec(grid=make_sampling_grid(sensor.timestamps_ns[:2], stride_ns=1, duration_ns=1))

    with pytest.raises(ValueError, match="unsupported decode_config"):
        next(sensor.sample(spec, policy=NearestTimestampPolicy()))


def test_camera_sensor_rejects_no_sampling_policy(synthetic_video: io.BytesIO) -> None:
    """CameraSensor accepts only NearestTimestampPolicy."""
    sensor = CameraSensor(synthetic_video.getvalue())
    spec = SamplingSpec(grid=make_sampling_grid(sensor.timestamps_ns[:2], stride_ns=1, duration_ns=1))

    with pytest.raises(TypeError, match="CameraSensor requires NearestTimestampPolicy"):
        next(sensor.sample(spec, policy=NoSamplingPolicy()))


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
        **_kwargs: object,
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
        **_kwargs: object,
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
    batches = list(sensor.sample(spec, policy=NearestTimestampPolicy()))

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
        **_kwargs: object,
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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        # Silence unused parameter warnings.
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
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

    batches = list(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))

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
        **_kwargs: object,
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
        **_kwargs: object,
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

    batch = next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))

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


def test_camera_sensor_populates_decoder_motion_vectors(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """CameraSensor should pass decoder-provided motion vectors through to CameraData."""
    index, metadata = _make_video_index_and_metadata(
        pts_ns=[100, 200],
        pts_stream=[10, 20],
        is_keyframe=[True, False],
        is_discard=[False, False],
        kf_pts_ns=[100],
        kf_pts_stream=[10],
    )
    motion_vectors = _make_motion_vector_data(2)

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: (  # noqa: ARG005
                np.zeros((2, metadata.height, metadata.width, 3), dtype=np.uint8),
                motion_vectors,
            ),
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

    batch = next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))

    assert batch.motion_vectors is motion_vectors


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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
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

    batch = next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))

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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
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

    batch = next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))

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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
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

    batch = next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))

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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
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

    batch = next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))

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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros(
                (sum(count for _, targets in decode_plan for _, count in targets), metadata.height, metadata.width, 3),
                dtype=np.uint8,
            ),
        )

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
    )

    sensor = CameraSensor(b"not-used", extrinsics=extrinsics)
    # Windows advance by 1000 ns from 150, so only the first holds a reference
    # timestamp; every later window is empty and yields the cached empty batch.
    grid = make_sampling_grid(
        timestamps_ns=np.array([150, 5_150], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batches = list(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))
    empty0 = batches[1]
    empty1 = sensor._get_empty_camera_data()

    assert len(empty0.align_timestamps_ns) == 0
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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
    ) -> _FakeDecoder:
        del source, stream_idx, config, stats
        return _FakeDecoder(
            time_base=index.time_base,
            decode_fn=lambda decode_plan: np.zeros(
                (sum(count for _, targets in decode_plan for _, count in targets), metadata.height, metadata.width, 3),
                dtype=np.uint8,
            ),
        )

    patch_camera_sensor_dependencies(
        make_index_and_metadata_fn=fake_make_index_and_metadata,
        decoder_open_fn=fake_decoder_open,
    )

    sensor = CameraSensor(b"not-used", intrinsics=intrinsics)
    # Windows advance by 1000 ns from 150, so only the first holds a reference
    # timestamp; every later window is empty and yields the cached empty batch.
    grid = make_sampling_grid(
        timestamps_ns=np.array([150, 5_150], dtype=np.int64),
        stride_ns=1_000,
        duration_ns=1_000,
    )

    batches = list(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))
    empty0 = batches[1]
    empty1 = sensor._get_empty_camera_data()

    assert len(empty0.align_timestamps_ns) == 0
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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
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
        assert isinstance(policy, NearestTimestampPolicy)
        raise AlignmentError(
            AlignmentFailureReason.TOLERANCE_EXCEEDED,
            "max_delta_ns=5 exceeded: max delta was 10 ns for grid=200, canonical=190",
            delta_ns=10,
            max_delta_ns=5,
        )

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
    )

    # The decoder context manager must not swallow or re-wrap what the sampler raised.
    # This says nothing about the tolerance rule itself — the fake supplied the
    # error. See test_camera_sensor_real_tolerance_failure_raises_alignment_error.
    with pytest.raises(AlignmentError, match="max_delta_ns=5 exceeded"):
        next(sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=5)))


def test_camera_sensor_real_tolerance_failure_raises_alignment_error(synthetic_video: io.BytesIO) -> None:
    """Drive a real decode through a genuine tolerance failure, with nothing patched.

    The fixture's frames land on 0, 33_333_333, 66_666_666 ns. The reference
    timestamp sits halfway between the first two, so whichever neighbour is
    selected is ~16.7 ms away — far outside a 1 ms tolerance. The window is wide
    enough to keep a neighbour eligible, so the failure is a genuine tolerance
    rejection rather than an empty batch.
    """
    sensor = CameraSensor(synthetic_video.getvalue())
    between_frames_ns = 16_666_666
    grid = make_sampling_grid(
        timestamps_ns=np.array([between_frames_ns, 50_000_000], dtype=np.int64),
        stride_ns=50_000_000,
        duration_ns=50_000_000,
    )

    with pytest.raises(AlignmentError) as caught:
        next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy(max_delta_ns=1_000_000)))

    assert caught.value.reason is AlignmentFailureReason.TOLERANCE_EXCEEDED
    assert caught.value.max_delta_ns == 1_000_000
    assert caught.value.delta_ns is not None
    assert caught.value.delta_ns > 1_000_000


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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
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

    batch = next(sensor.sample(SamplingSpec(grid=grid), policy=NearestTimestampPolicy()))

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
        **_kwargs: object,
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
    assert sensor.has_bframes is False


@pytest.mark.env("default")
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
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    def fake_gpu_decoder_open(
        source: object,
        stream_idx: int = 0,
        config: object = None,
        stats: object = None,
        **_kwargs: object,
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

    batch = next(sensor.sample(spec, policy=NearestTimestampPolicy()))

    assert len(gpu_open_calls) == 1
    assert gpu_open_calls[0][0] == b"not-used"
    assert gpu_open_calls[0][1] == 0
    assert gpu_open_calls[0][2] is decode_config
    assert gpu_open_calls[0][3] is None
    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100, 200], dtype=np.int64))


# --- stream_timestamps -------------------------------------------------------


class _PoisonSource:
    """A video source that fails on any access.

    The ``VideoIndex`` is stubbed in via ``patch_camera_sensor_dependencies``, so
    a sensor built over this source never touches it during construction. It is
    reached only if something tries to *decode* frames — which would open the
    source through some decoder — so any access fails the test, independent of
    which concrete decoder class does the opening.
    """

    def __getattr__(self, name: str) -> object:
        msg = f"the video source must not be accessed (attempted .{name})"
        raise AssertionError(msg)


def _linear_camera_sensor(
    patch_camera_sensor_dependencies: Callable[..., None],
    *,
    pts_ns: list[int],
    is_discard: list[bool] | None = None,
    source: object = b"not-used",
) -> CameraSensor:
    """Build a CameraSensor over a fake index with a strictly-increasing timeline."""
    n = len(pts_ns)
    discard = [False] * n if is_discard is None else is_discard
    index, metadata = _make_video_index_and_metadata(
        pts_ns=pts_ns,
        pts_stream=[i + 1 for i in range(n)],
        is_keyframe=[i == 0 for i in range(n)],
        is_discard=discard,
        kf_pts_ns=[pts_ns[0]],
        kf_pts_stream=[1],
    )

    def fake_make_index_and_metadata(
        source: object,
        stream_idx: int = 0,
        index_method: object = None,
        **_kwargs: object,
    ) -> tuple[VideoIndex, VideoMetadata]:
        del source, stream_idx, index_method
        return index, metadata

    patch_camera_sensor_dependencies(make_index_and_metadata_fn=fake_make_index_and_metadata)
    return CameraSensor(source)


def _count_decoded_frames(video_bytes: bytes) -> int:
    """Count displayable frames by decoding the clip independently of the index."""
    count = 0
    with av.open(io.BytesIO(video_bytes)) as container:
        for _frame in container.decode(video=0):
            count += 1
    return count


def test_stream_timestamps_single_batch_covers_full_timeline(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """batch_size == 0 yields one batch equal to the full presentation timeline."""
    sensor = _linear_camera_sensor(patch_camera_sensor_dependencies, pts_ns=[100, 200, 300, 400, 500])

    batches = list(sensor.stream_timestamps())

    assert len(batches) == 1
    assert batches[0].dtype == np.int64
    np.testing.assert_array_equal(batches[0], sensor.timestamps_ns)
    assert bool(np.all(np.diff(batches[0]) > 0))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 8])
def test_stream_timestamps_batches_tile_the_timeline(
    patch_camera_sensor_dependencies: Callable[..., None],
    batch_size: int,
) -> None:
    """batch_size > 0 yields consecutive, non-overlapping batches whose union is the timeline."""
    sensor = _linear_camera_sensor(patch_camera_sensor_dependencies, pts_ns=[100 * (i + 1) for i in range(7)])

    batches = list(sensor.stream_timestamps(batch_size))

    for batch in batches[:-1]:
        assert len(batch) == batch_size
    assert 0 < len(batches[-1]) <= batch_size
    np.testing.assert_array_equal(np.concatenate(batches), sensor.timestamps_ns)


def test_stream_timestamps_excludes_discarded_packets(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """The streamed timeline is the displayable presentation timeline (discards removed)."""
    sensor = _linear_camera_sensor(
        patch_camera_sensor_dependencies,
        pts_ns=[100, 200, 300, 400],
        is_discard=[False, True, False, False],
    )

    (timeline,) = list(sensor.stream_timestamps())

    np.testing.assert_array_equal(timeline, np.array([100, 300, 400], dtype=np.int64))
    np.testing.assert_array_equal(timeline, sensor.timestamps_ns)


def test_stream_timestamps_rejects_negative_batch_size(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """A negative batch_size is rejected eagerly."""
    sensor = _linear_camera_sensor(patch_camera_sensor_dependencies, pts_ns=[100, 200, 300])

    with pytest.raises(ValueError, match="batch_size must be non-negative"):
        sensor.stream_timestamps(-1)


def test_stream_timestamps_reads_index_without_decoding(
    patch_camera_sensor_dependencies: Callable[..., None],
) -> None:
    """stream_timestamps reads the index only: it never opens or reads the source.

    The sensor is built over a source that raises on any access, so if the
    method decoded frames — the outcome forbidden by the ticket — reaching for
    the source would fail. Success means the timeline was produced from the
    index alone, whatever decoder class a decode path would have used.
    """
    sensor = _linear_camera_sensor(
        patch_camera_sensor_dependencies,
        pts_ns=[100, 200, 300, 400],
        source=_PoisonSource(),
    )

    batches = list(sensor.stream_timestamps(2))

    np.testing.assert_array_equal(np.concatenate(batches), sensor.timestamps_ns)


def test_stream_timestamps_bframe_clip_is_monotonic_presentation_order(
    h264_video: Callable[..., bytes],
) -> None:
    """B-frame content: AUTO -> FULL_DEMUX yields a strictly-increasing timeline of frame_count length."""
    video_bytes = h264_video(bframes=1)
    sensor = CameraSensor(video_bytes)

    # has_bframes True confirms the AUTO index method selected FULL_DEMUX for this clip.
    assert sensor.has_bframes is True

    (timeline,) = list(sensor.stream_timestamps())

    assert timeline.dtype == np.int64
    assert bool(np.all(np.diff(timeline) > 0))
    assert len(timeline) == _count_decoded_frames(video_bytes)
    np.testing.assert_array_equal(timeline, sensor.timestamps_ns)

    # Batched reads over the same clip tile the identical timeline.
    np.testing.assert_array_equal(np.concatenate(list(sensor.stream_timestamps(64))), timeline)
