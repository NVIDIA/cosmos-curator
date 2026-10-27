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
"""Tests for the forward-only streaming camera sensor."""

import gc
import hashlib
import itertools
import pathlib
import struct
from collections.abc import Iterator
from fractions import Fraction
from typing import Any
from unittest.mock import Mock

import attrs
import av
import numpy as np
import psutil
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata, UnsupportedEditListError
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.group import SensorGroup
from cosmos_curator.core.sensors.sensors.streaming_camera_sensor import (
    IndexRecordUnusableError,
    StreamingCameraSensor,
    stream_camera_windows,
)
from cosmos_curator.core.sensors.types.types import TimestampSource, VideoFrameAlignmentField
from cosmos_curator.core.sensors.utils.video import (
    StreamingDecodePlan,
    StreamingVideoDecoder,
    iter_video_frames,
)
from cosmos_curator.next.recipes.mp4_index.contracts import FRAME_INDEX_SCHEMA

_DATA = pathlib.Path(__file__).resolve().parents[3] / "pipelines" / "video" / "data"
_CLIP = _DATA / "test_clip_10s.mp4"
_BFRAME_CLIP = _DATA / "test_clip_10s_bframes.mp4"
_LONG_CLIP = _DATA / "test_video_30s.mp4"

# 240 frames of 854x480 at 24 fps, so ~41.67 ms of source spacing and ~1.2 MB per RGB frame.
_SOURCE_FRAME_COUNT = 240
_LONG_CLIP_FRAME_COUNT = 720
_FRAME_BYTES = 854 * 480 * 3
_ONE_MS_NS = 1_000_000

_ONE_SECOND_NS = 1_000_000_000
# A recorder's uptime clock, taken from real footage: nowhere near a container's zero.
_DEVICE_ORIGIN_NS = 3_517_693_838_000
_NS_PER_SECOND = 1_000_000_000


def _sensor_with_eof(source: pathlib.Path, timeline: np.ndarray) -> StreamingCameraSensor:
    """Build a supplied index with synthetic EOF values for sampling tests."""
    indexed = StreamingCameraSensor(source)
    metadata, frame_index = indexed.metadata, indexed.frame_index
    if len(timeline) != len(frame_index):
        metadata = attrs.evolve(metadata, frame_count=len(timeline))
        frame_index = _frame_index(_frame_rows(len(timeline)), metadata)
    inverse_order = np.argsort(frame_index.pts_sort_order)
    frame_index = attrs.evolve(frame_index, eof_ns=timeline[inverse_order])
    return StreamingCameraSensor(
        source, metadata=metadata, frame_index=frame_index, alignment_field=VideoFrameAlignmentField.EOF_NS
    )


def _whole_clip(hz: float, count: int, origin_ns: int = 0) -> SamplingSpec:
    """One window holding *count* reference timestamps at ``hz``, starting at ``origin_ns``.

    ``round(k * step)`` rather than ``k * round(step)``, matching make_ts_grid: a
    step that is not a whole number of nanoseconds drifts a nanosecond per frame
    the other way, until the last reference falls past the last frame.
    """
    step = _NS_PER_SECOND / hz
    timestamps = np.array([origin_ns + round(index * step) for index in range(count)], dtype=np.int64)
    span_ns = int(timestamps[-1]) + 1 - origin_ns
    return SamplingSpec(
        grid=SamplingGrid(origin_ns, origin_ns + span_ns, timestamps, stride_ns=span_ns, duration_ns=span_ns)
    )


def _walk(
    sensor: StreamingCameraSensor,
    hz: float,
    max_delta_ns: int,
    count: int = _SOURCE_FRAME_COUNT,
    origin_ns: int = 0,
) -> list:
    """Drain every window of a whole-clip spec into one list of rows."""
    policy = NearestTimestampPolicy(max_delta_ns=max_delta_ns)
    return [
        row for payload in sensor.sample(_whole_clip(hz, count, origin_ns), policy=policy) for row in payload.frames()
    ]


def _starting_late(source: pathlib.Path, destination: pathlib.Path, offset_ns: int) -> pathlib.Path:
    """Copy a clip's packets forward in time, so its first PTS is not zero.

    Real recordings do this; the encoder cannot be made to, because the MP4 muxer
    normalizes an encode back to zero. Remuxing preserves the shift, which is the
    only way to build the case that tells an origin apart from an offset.
    """
    with av.open(str(source)) as reader, av.open(str(destination), mode="w") as writer:
        stream = reader.streams.video[0]
        offset = int(offset_ns * stream.time_base.denominator / (stream.time_base.numerator * _NS_PER_SECOND))
        output_stream = writer.add_stream_from_template(stream)
        for packet in reader.demux(stream):
            if packet.pts is None:
                continue
            packet.pts += offset
            packet.dts = (packet.dts or 0) + offset
            packet.stream = output_stream
            writer.mux(packet)
    return destination


def test_walks_the_clip_at_the_requested_rate() -> None:
    """Reference timestamps start at the first frame's PTS and step at 1/hz."""
    sensor = _sensor_with_eof(_CLIP, _timeline(_CLIP))

    rows = [(align_ns, pts_ns) for align_ns, pts_ns, _ in _walk(sensor, 24.0, _ONE_MS_NS)]

    align_ns = [row[0] for row in rows]
    pts_ns = [row[1] for row in rows]
    # References are ``origin + round(k * step)``, so consecutive gaps alternate
    # between the two integers bracketing a step that is not a whole number of
    # nanoseconds. Pinning a single gap would pin a rounding artifact instead of
    # the rate, and would drift a nanosecond per frame away from the true one.
    step_ns = 1e9 / 24.0
    assert align_ns[0] == pts_ns[0] == 0
    assert {b - a for a, b in itertools.pairwise(align_ns)} <= {int(step_ns), int(step_ns) + 1}
    assert align_ns[-1] == round((len(align_ns) - 1) * step_ns)
    # References are ``round(k * step)``, matching make_ts_grid, so they track the
    # true frame spacing instead of drifting a nanosecond per frame away from it.
    # Every source frame therefore gets a reference.
    assert len(rows) == _SOURCE_FRAME_COUNT
    assert max(abs(a - p) for a, p in rows) < 1000


def test_bframe_clip_builds_its_index_lazily() -> None:
    """A video can be decoded without supplying a prebuilt index."""
    sensor = StreamingCameraSensor(_BFRAME_CLIP, alignment_field=VideoFrameAlignmentField.EDITED_PTS_NS)

    pts_ns = [pts for _, pts, _ in _walk(sensor, 24.0, _ONE_MS_NS)]

    assert len(pts_ns) == _SOURCE_FRAME_COUNT
    assert pts_ns == sorted(pts_ns)
    assert len(set(pts_ns)) == _SOURCE_FRAME_COUNT


@pytest.mark.parametrize("clip", [_CLIP, _BFRAME_CLIP], ids=["no-bframes", "bframes"])
@pytest.mark.parametrize("windows_per_clip", [2, 5])
def test_window_count_does_not_change_selected_timestamps_or_pixels(clip: pathlib.Path, windows_per_clip: int) -> None:
    """Window boundaries group references without restricting eligible source frames."""
    start_ns = _DEVICE_ORIGIN_NS + 20_000_000
    duration_ns = 3 * _ONE_SECOND_NS
    references_ns = np.array([start_ns + round(i * _ONE_SECOND_NS / 30) for i in range(90)], dtype=np.int64)
    signatures = []
    for window_count in (1, windows_per_clip):
        width_ns = duration_ns // window_count
        spec = SamplingSpec(
            grid=SamplingGrid(start_ns, start_ns + duration_ns, references_ns, stride_ns=width_ns, duration_ns=width_ns)
        )
        sensor = _sensor_with_eof(clip, _timeline(clip, _DEVICE_ORIGIN_NS))
        payloads = list(sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=21_000_000)))

        assert len(payloads) == window_count
        assert [len(payload.align_timestamps_ns) for payload in payloads] == [90 // window_count] * window_count
        if window_count > 1:
            # The next source frame is outside tolerance at these edges; the preceding frame must remain eligible.
            assert any(payload.sensor_timestamps_ns[0] < payload.align_timestamps_ns[0] for payload in payloads[1:])
        rows = [
            (reference_ns, source_ns, hashlib.sha256(frame.tobytes()).digest())
            for payload in payloads
            for reference_ns, source_ns, frame in payload.frames()
        ]
        assert [reference_ns for reference_ns, _, _ in rows] == references_ns.tolist()
        signatures.append(rows)

    assert signatures[0] == signatures[1]


@pytest.mark.parametrize("window_count", [1, 5])
def test_window_payloads_materialize_only_the_requested_frames(
    monkeypatch: pytest.MonkeyPatch, window_count: int
) -> None:
    """Requesting windows decodes nothing; pulling a row advances only the shared cursor."""
    decoded_count = 0
    timestamps_ns = np.arange(30, dtype=np.int64) * 10

    def decoded_frames(_decoder: StreamingVideoDecoder, plan: StreamingDecodePlan) -> Iterator[tuple[int, np.ndarray]]:
        nonlocal decoded_count
        for index, count in zip(plan.frame_indices, plan.counts, strict=True):
            decoded_count += 1
            for _ in range(int(count)):
                yield int(index), np.full((1, 1, 3), index, dtype=np.uint8)

    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.streaming_camera_sensor.execute_streaming_decode_plan", decoded_frames
    )
    sensor = _sensor_with_eof(_CLIP, timestamps_ns)
    width_ns = 300 // window_count
    spec = SamplingSpec(grid=SamplingGrid(0, 300, timestamps_ns, stride_ns=width_ns, duration_ns=width_ns))
    payloads = list(sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=0)))

    assert len(payloads) == window_count
    assert decoded_count == 0
    rows = (row for payload in payloads for row in payload.frames())
    row_count = 0
    for index, (reference_ns, source_ns, frame) in enumerate(rows):
        assert decoded_count == index + 1
        assert reference_ns == source_ns == timestamps_ns[index]
        assert frame[0, 0, 0] == index
        row_count += 1
    assert row_count == len(timestamps_ns)


def test_one_source_frame_serves_several_reference_timestamps() -> None:
    """Sampling faster than the source rate repeats the nearest frame, without re-decoding it."""
    sensor = _sensor_with_eof(_CLIP, _timeline(_CLIP))

    # Four references per source frame, stopping short of the last one: the grid
    # decides where the timeline ends now, and a reference past the final frame is
    # a tolerance failure rather than the end of the walk.
    references = 4 * (_SOURCE_FRAME_COUNT - 1)
    rows = [(pts_ns, frame) for _, pts_ns, frame in _walk(sensor, 96.0, 21_000_000, references)]

    pts_ns = [pts for pts, _ in rows]
    assert pts_ns == sorted(pts_ns)
    assert len(rows) == references
    assert len(set(pts_ns)) == _SOURCE_FRAME_COUNT
    # A repeated reference timestamp reuses the frame it already holds.
    repeats = [(prev, cur) for prev, cur in itertools.pairwise(rows) if prev[0] == cur[0]]
    assert len(repeats) > 0
    assert all(prev[1] is cur[1] for prev, cur in repeats)


def test_frame_farther_away_than_the_tolerance_raises() -> None:
    """A reference timestamp with no frame within ``max_delta_ns`` is an error."""
    sensor = _sensor_with_eof(_CLIP, _timeline(_CLIP))

    with pytest.raises(ValueError, match=r"max_delta_ns=1000000 exceeded"):
        for _ in _walk(sensor, 1000.0, _ONE_MS_NS, 100):
            pass


def test_peak_memory_stays_flat_over_a_long_walk() -> None:
    """Resident memory must not grow with stream length: consume frames without retaining them.

    Walks 720 frames of 854x480 in one pass -- 0.9 GB of pixels if any of it
    were retained -- and compares peak RSS against a baseline taken once the
    decoder is warm. The threshold sits above RSS jitter rather than tight
    against it; the failure this catches is linear growth, which lands two
    orders of magnitude past it.
    """
    process = psutil.Process()
    warmup_rows = 60
    sensor = _sensor_with_eof(_LONG_CLIP, _timeline(_LONG_CLIP))
    rows = 0
    baseline_rss = 0
    peak_rss = 0

    spec = _whole_clip(24.0, _LONG_CLIP_FRAME_COUNT)
    payloads = sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=_ONE_MS_NS))
    for _align_ns, _pts_ns, frame in (row for payload in payloads for row in payload.frames()):
        assert frame.shape == (480, 854, 3)
        rows += 1
        if rows == warmup_rows:
            gc.collect()
            baseline_rss = process.memory_info().rss
        elif rows > warmup_rows:
            peak_rss = max(peak_rss, process.memory_info().rss)

    assert rows == _LONG_CLIP_FRAME_COUNT
    growth_frames = (peak_rss - baseline_rss) / _FRAME_BYTES
    assert growth_frames < 8, f"RSS grew by {growth_frames:.1f} frames over {rows} decoded frames"


def _grid(start_ns: int, *, stride_ns: int, duration_ns: int, count: int, hz: float) -> SamplingSpec:
    """Build a spec whose windows tile forward from ``start_ns``."""
    step = round(1e9 / hz)
    windows_end = start_ns + stride_ns * count
    timestamps = np.arange(start_ns, windows_end, step, dtype=np.int64)
    return SamplingSpec(
        grid=SamplingGrid(start_ns, windows_end, timestamps, stride_ns=stride_ns, duration_ns=duration_ns)
    )


def test_sample_yields_one_payload_per_window() -> None:
    """SensorGroup advances every sensor once per window, so a payload per window is the contract."""
    sensor = _sensor_with_eof(_CLIP, _timeline(_CLIP))
    spec = _grid(0, stride_ns=_ONE_SECOND_NS, duration_ns=_ONE_SECOND_NS, count=3, hz=24.0)

    payloads = list(sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=_ONE_MS_NS)))

    assert len(payloads) == 3
    for payload, window in zip(payloads, spec.grid, strict=True):
        rows = list(payload.frames())
        assert [row[0] for row in rows] == window.timestamps_ns.tolist()


def test_an_empty_window_still_yields_a_payload() -> None:
    """A window with no reference timestamps must not desynchronise the group.

    SensorGroup walks every sensor in lockstep; a sensor that skipped a window
    would hand the next window's frames to the wrong one.
    """
    sensor = _sensor_with_eof(_CLIP, _timeline(_CLIP))
    step = round(1e9 / 24.0)
    # Second window covers a stretch the grid places no timestamps in.
    timestamps = np.array([0, step, 3 * _ONE_SECOND_NS], dtype=np.int64)
    spec = SamplingSpec(
        grid=SamplingGrid(0, 4 * _ONE_SECOND_NS, timestamps, stride_ns=_ONE_SECOND_NS, duration_ns=_ONE_SECOND_NS)
    )

    payloads = list(sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS)))

    assert len(payloads) == 4
    assert list(payloads[1].frames()) == []


def test_overlapping_windows_are_rejected() -> None:
    """A forward-only walk cannot serve a window it has already passed.

    Buffering the overlap would reintroduce the memory this sensor exists to
    avoid, so overlap is refused rather than absorbed. McapCameraSensor refuses
    it for the same reason.
    """
    sensor = _sensor_with_eof(_CLIP, _timeline(_CLIP))
    spec = _grid(0, stride_ns=_ONE_SECOND_NS // 2, duration_ns=_ONE_SECOND_NS, count=3, hz=24.0)

    def drain() -> None:
        for payload in sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=_ONE_MS_NS)):
            list(payload.frames())

    with pytest.raises(ValueError, match="non-overlapping"):
        drain()


def test_a_late_starting_container_reports_its_own_first_pts() -> None:
    """The fixture must really start late, or the origin tests below prove nothing."""
    with av.open(str(_CLIP)) as reader:
        native_start = int(reader.streams.video[0].start_time * reader.streams.video[0].time_base * _NS_PER_SECOND)

    assert native_start == 0


def test_unsupported_edits_raise_when_a_lazy_index_is_built(tmp_path: pathlib.Path) -> None:
    """A multi-entry edit introduced by remuxing is rejected by the frame-index builder."""
    late = _starting_late(_CLIP, tmp_path / "late.mp4", _ONE_SECOND_NS // 2)
    sensor = StreamingCameraSensor(late)
    with pytest.raises(UnsupportedEditListError, match="entries"):
        _ = sensor.frame_index


def test_origin_moves_the_timeline_without_reshaping_it() -> None:
    """Only the origin moves: the gaps between frames are the recording's own."""
    native = [pts for _, pts, _ in _walk(_sensor_with_eof(_CLIP, _timeline(_CLIP)), 24.0, _ONE_MS_NS)]

    shifted = [
        pts
        for _, pts, _ in _walk(
            _sensor_with_eof(_CLIP, _timeline(_CLIP, _DEVICE_ORIGIN_NS)),
            24.0,
            _ONE_MS_NS,
            origin_ns=_DEVICE_ORIGIN_NS,
        )
    ]

    assert shifted == [pts + _DEVICE_ORIGIN_NS - native[0] for pts in native]


def _timeline(clip: pathlib.Path, origin_ns: int = 0) -> np.ndarray:
    """Return synthetic sensor timestamps spaced by this clip's frame intervals.

    Anchored at ``origin_ns`` and spaced by the container's own intervals, which
    is the shape of the real artifact: when recording started is a fact about the
    recorder, and where the container puts its zero is not.
    """
    with av.open(str(clip)) as reader:
        stream = reader.streams.video[0]
        pts = sorted(
            int(packet.pts * stream.time_base * _NS_PER_SECOND)
            for packet in reader.demux(stream)
            if packet.pts is not None
        )
    return np.array([origin_ns + value - pts[0] for value in pts], dtype=np.int64)


def test_it_is_a_sensor_group_member() -> None:
    """A sensor that cannot be driven by SensorGroup is not a sensor in this library.

    Lockstep alignment across modalities is what the group exists for, and it is
    the reason this serves one payload per window rather than skipping the empty
    ones. That justification is only worth anything if the group can actually
    drive it, so this drives it.
    """
    timeline = _timeline(_CLIP, _DEVICE_ORIGIN_NS)
    sensor = _sensor_with_eof(_CLIP, timeline)
    group = SensorGroup({"camera": sensor})

    assert group.start_ns == _DEVICE_ORIGIN_NS
    assert sensor.end_ns == int(timeline[-1])
    assert sensor.supports_sampling_policy(NearestTimestampPolicy(max_delta_ns=_ONE_MS_NS))

    spec = _whole_clip(24.0, _SOURCE_FRAME_COUNT, _DEVICE_ORIGIN_NS)
    frames = list(group.sample(spec, policies={"camera": NearestTimestampPolicy(max_delta_ns=_ONE_MS_NS)}))

    assert len(frames) == 1
    rows = list(frames[0]["camera"].frames())
    assert len(rows) == _SOURCE_FRAME_COUNT
    assert rows[0][0] == _DEVICE_ORIGIN_NS


def test_a_group_gets_one_payload_for_every_window_it_asks_for() -> None:
    """Lockstep means a window always produces a payload, even one covering nothing."""
    timeline = _timeline(_CLIP, _DEVICE_ORIGIN_NS)
    sensor = _sensor_with_eof(_CLIP, timeline)
    group = SensorGroup({"camera": sensor})
    step = _NS_PER_SECOND // 4
    timestamps = np.array([_DEVICE_ORIGIN_NS + index * step for index in range(8)], dtype=np.int64)
    spec = SamplingSpec(
        grid=SamplingGrid(
            _DEVICE_ORIGIN_NS, _DEVICE_ORIGIN_NS + 8 * step, timestamps, stride_ns=2 * step, duration_ns=2 * step
        )
    )

    frames = list(group.sample(spec, policies={"camera": NearestTimestampPolicy(max_delta_ns=_ONE_MS_NS)}))

    assert len(frames) == 4
    assert all(len(list(frame["camera"].frames())) == 2 for frame in frames)


def _two_windows(origin_ns: int) -> SamplingSpec:
    """Return a grid of two adjacent windows, so payload order can be got wrong."""
    step = _NS_PER_SECOND // 4
    timestamps = np.array([origin_ns + index * step for index in range(4)], dtype=np.int64)
    return SamplingSpec(
        grid=SamplingGrid(origin_ns, origin_ns + 4 * step, timestamps, stride_ns=2 * step, duration_ns=2 * step)
    )


def test_payloads_taken_out_of_order_raise_rather_than_serve_the_wrong_frames() -> None:
    """One walk backs every payload, so an earlier window cannot be served after a later one.

    Retaining the payloads is legitimate -- they are yielded before they are
    consumed -- but the decode position is shared and cannot rewind. Serving the
    earlier window from the later position would pick real frames for the wrong
    timestamps, which a tolerance check catches only by luck.
    """
    sensor = _sensor_with_eof(_CLIP, _timeline(_CLIP))
    payloads = list(sensor.sample(_two_windows(0), policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS)))

    assert len(list(payloads[1].frames())) == 2

    with pytest.raises(RuntimeError, match="order"):
        list(payloads[0].frames())


def test_a_payload_cannot_be_resumed_once_a_later_one_has_started() -> None:
    """Interleaving is the same violation, caught at the row rather than at the start."""
    sensor = _sensor_with_eof(_CLIP, _timeline(_CLIP))
    payloads = list(sensor.sample(_two_windows(0), policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS)))
    first = payloads[0].frames()
    next(first)
    next(payloads[1].frames())

    with pytest.raises(RuntimeError, match="order"):
        next(first)


def _drain(sensor: StreamingCameraSensor, spec: SamplingSpec) -> list:
    """Consume every payload of *spec*, in order."""
    policy = NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS)
    return [row for payload in sensor.sample(spec, policy=policy) for row in payload.frames()]


def test_a_timeline_must_be_one_dimensional() -> None:
    """Caught at construction, not at the first property that tries to read a scalar."""
    with pytest.raises(ValueError, match="must be 1-D"):
        _sensor_with_eof(_CLIP, np.array([[1, 2], [3, 4]], dtype=np.int64))


def test_an_ascending_timeline_is_not_rejected_for_being_wide() -> None:
    """Adjacent values are compared, not differenced: a gap can exceed what int64 holds."""
    limits = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)

    assert _sensor_with_eof(_CLIP, limits).start_ns == int(np.iinfo(np.int64).min)


def test_a_timeline_that_rounds_differently_from_the_container_still_serves() -> None:
    """Uniformly spaced sensor timestamps and container PTS round differently.

    A container's time base rarely divides a nanosecond evenly: at 1/12288 this
    clip puts 80 of its 240 frames one nanosecond off where a sensor timeline at
    the same rate would. That is two representations of one instant, not a
    timeline describing a different recording.
    """
    step = _NS_PER_SECOND / 24.0
    timeline = np.array([round(index * step) for index in range(_SOURCE_FRAME_COUNT)], dtype=np.int64)

    rows = _walk(_sensor_with_eof(_CLIP, timeline), 24.0, _ONE_MS_NS)

    assert len(rows) == _SOURCE_FRAME_COUNT


def test_rows_report_the_timeline_not_the_container() -> None:
    """The timeline is the sensor's clock; a container's PTS is a different measurement.

    An mp4 written at a declared rate spaces its frames perfectly evenly. The
    recorder's timeline does not: on real 30 fps footage the intervals run 33.306
    to 33.361 ms, and the container reports 33.3333 for every one of them. The
    container has thrown that away, so reporting it would assert an evenness the
    recording never had.
    """
    timeline = _timeline(_CLIP, _DEVICE_ORIGIN_NS)

    rows = _walk(_sensor_with_eof(_CLIP, timeline), 24.0, _ONE_MS_NS, origin_ns=_DEVICE_ORIGIN_NS)

    assert [pts for _, pts, _ in rows] == [int(value) for value in timeline]


def test_a_container_with_fewer_frames_than_the_timeline_raises() -> None:
    """A recording that ends before its timeline does must not serve the final frame forever.

    Without this check the container's clean EOF is indistinguishable from a
    truncated or mismatched one: later references would silently reuse the last
    decoded frame while still publishing the timeline's (unobserved) timestamps
    for them.
    """
    native = _timeline(_CLIP)
    longer = np.concatenate([native, [int(native[-1]) + 1]])

    payloads = stream_camera_windows(
        iter_video_frames(_CLIP),
        longer,
        _whole_clip(24.0, len(longer)),
        policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS),
    )
    with pytest.raises(ValueError, match="container ended"):
        [row for payload in payloads for row in payload.frames()]


def test_a_container_with_more_frames_than_the_timeline_raises() -> None:
    """The reverse mismatch: a recording that outlives its own declared timeline."""
    native = _timeline(_CLIP)
    shorter = native[:-1]

    payloads = stream_camera_windows(
        iter_video_frames(_CLIP),
        shorter,
        _whole_clip(24.0, len(shorter)),
        policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS),
    )
    with pytest.raises(ValueError, match="holds more frames"):
        [row for payload in payloads for row in payload.frames()]


def test_a_missed_exposure_is_served_by_repeating_its_neighbour() -> None:
    """A camera that skips a frame still yields a full clip, with a frame serving twice.

    Real footage: one camera on a rig waited two frame periods instead of one,
    18 seconds in. It still wrote as many frames as everyone else, and its
    container -- written at a declared constant rate -- spaced them evenly with
    no gap. Only the timeline records the pause.

    Nothing about that is unservable. The reference landing in the gap is nearer
    to one of the two frames bracketing it than to any other, so that frame
    serves twice, which is what nearest-timestamp sampling is for.
    """
    step = _NS_PER_SECOND // 24
    missed = 80
    # One doubled interval, as a missed exposure looks: the camera fired at 79
    # and again at 81's moment, so nothing was recorded for 80's.
    gaps = [0] + [step if index != missed else 2 * step for index in range(1, _SOURCE_FRAME_COUNT)]
    timeline = np.cumsum(np.array(gaps, dtype=np.int64))
    sensor = _sensor_with_eof(_CLIP, timeline)
    references = np.array([round(index * step) for index in range(_SOURCE_FRAME_COUNT)], dtype=np.int64)
    span = int(references[-1]) + 1
    spec = SamplingSpec(grid=SamplingGrid(0, span, references, stride_ns=span, duration_ns=span))

    rows = [
        row
        for payload in sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=step))
        for row in payload.frames()
    ]

    served = [pts for _, pts, _ in rows]
    assert len(rows) == _SOURCE_FRAME_COUNT
    assert len(set(served)) == _SOURCE_FRAME_COUNT - 1

    # Not merely that something repeated: the reference with no observation of
    # its own is the one inside the gap, and it is served by whichever of the
    # two frames bracketing that gap is nearer. Repeating any other frame would
    # satisfy the counts above while putting the wrong pixels at that instant.
    doubled = [index for index in range(1, len(served)) if served[index] == served[index - 1]]
    assert doubled == [missed]
    bracketing = {int(timeline[missed - 1]), int(timeline[missed])}
    assert served[missed] in bracketing
    assert abs(served[missed] - int(references[missed])) == min(
        abs(candidate - int(references[missed])) for candidate in bracketing
    )


#: Ticks per second for the index fixtures. Deliberately not a divisor of 1e9 -- one tick
#: is 81380.2... ns -- so the tick-to-nanosecond conversion has to actually happen for
#: these timestamps to come out right, rather than passing under an evenly dividing scale.
_INDEX_TIME_BASE_DEN = 12288

#: One frame interval at the index fixtures' nominal 30 fps, in ticks.
_INDEX_FRAME_TICKS = _INDEX_TIME_BASE_DEN // 30


def _metadata_row(**overrides: object) -> dict[str, Any]:
    """One VIDEO_METADATA_SCHEMA row, with the fields these tests vary pulled to the front."""
    row: dict[str, Any] = {
        "track_id": 1,
        "codec_name": "h264",
        "extradata": b"\x01\x64\x00\x1f",
        "time_base_num": 1,
        "time_base_den": _INDEX_TIME_BASE_DEN,
        "frame_count": 3,
        "keyframe_count": 1,
        "width": 640,
        "height": 480,
        "avc_profile_idc": 100,
        "avc_constraint_flags": 0,
        "avc_level_idc": 31,
        "nal_length_size": 4,
        "chroma_format_idc": 1,
        "bit_depth_luma_minus8": 0,
        "bit_depth_chroma_minus8": 0,
        "buffer_size_db": None,
        "max_bitrate_bps": None,
        "avg_bitrate_bps": None,
        "movie_creation_time": 0,
        "movie_modification_time": 0,
        "track_creation_time": 0,
        "track_modification_time": 0,
        "track_matrix": [0] * 9,
        "track_display_width": 0,
        "track_display_height": 0,
        "elst_bytes": None,
        "avg_frame_rate_num": 30,
        "avg_frame_rate_den": 1,
        "bit_rate_bps": 1000,
        "duration_ns": 1_000_000_000,
        "encoder_tag": None,
    }
    row.update(overrides)
    return row


def _frame_rows(count: int = 3, *, composition_offset_ticks: int | None = None) -> list[dict[str, Any]]:
    """``count`` evenly spaced frames in decode order, all sharing one composition offset."""
    return [
        {
            "dts_ticks": position * _INDEX_FRAME_TICKS,
            "composition_offset_ticks": composition_offset_ticks,
            "is_keyframe": position == 0,
            "offset": position * 100,
            "size": 100,
        }
        for position in range(count)
    ]


def _frame_index(rows: list[dict[str, Any]], metadata: Mp4Metadata) -> Mp4FrameIndex:
    return Mp4FrameIndex.from_frame_index_table(
        pa.Table.from_pylist(rows, schema=FRAME_INDEX_SCHEMA),
        time_base_num=metadata.time_base_num,
        time_base_den=metadata.time_base_den,
        elst_bytes=metadata.elst_bytes,
    )


def _sensor_from_rows(rows: list[dict[str, Any]] | None = None, **metadata_overrides: object) -> StreamingCameraSensor:
    """Build an index-backed sensor over *rows*, with no media behind it."""
    frames = _frame_rows() if rows is None else rows
    metadata = Mp4Metadata.from_row(_metadata_row(frame_count=len(frames), **metadata_overrides))
    return StreamingCameraSensor(
        None,
        frame_index=_frame_index(frames, metadata),
        metadata=metadata,
        alignment_field=VideoFrameAlignmentField.DTS_NS,
    )


def _indexed_ns(tick: int) -> int:
    return tick * _NS_PER_SECOND // _INDEX_TIME_BASE_DEN


def test_an_index_timeline_is_the_decode_timestamps_in_nanoseconds() -> None:
    """The reported timeline is dts_ticks on the track's time base, not raw ticks."""
    sensor = _sensor_from_rows()

    assert sensor.timestamps_ns.dtype == np.int64
    assert sensor.timestamps_ns.tolist() == [_indexed_ns(position * _INDEX_FRAME_TICKS) for position in range(3)]
    assert sensor.start_ns == 0
    assert sensor.end_ns == _indexed_ns(2 * _INDEX_FRAME_TICKS)


def test_an_index_timeline_is_reported_as_decode_time() -> None:
    """Not merely "a container clock": decode order ascends, so an in-order finding is a tautology."""
    assert _sensor_from_rows().timestamp_source is TimestampSource.CONTAINER_DTS


def test_scalar_container_facts_come_off_the_metadata_struct() -> None:
    """Codec and nominal rate are read from the row rather than inferred from the frames."""
    sensor = _sensor_from_rows(codec_name="hevc", avg_frame_rate_num=30000, avg_frame_rate_den=1001)

    assert sensor.codec_name == "hevc"
    assert sensor.avg_frame_rate == Fraction(30000, 1001)


def test_a_declared_rate_of_zero_stays_zero_rather_than_raising() -> None:
    """Fraction(0) is the library's "no usable header rate" sentinel, which makes rate metrics skip."""
    assert _sensor_from_rows(avg_frame_rate_num=0, avg_frame_rate_den=1).avg_frame_rate == Fraction(0)


def test_a_zero_denominator_degrades_to_the_same_sentinel() -> None:
    """A denominator the writer cannot actually emit still skips rather than raising ZeroDivisionError."""
    assert _sensor_from_rows(avg_frame_rate_den=0).avg_frame_rate == Fraction(0)


def test_an_absent_composition_offset_column_means_no_reordering() -> None:
    """All-null offsets mean the ctts box is absent, so the container signals no reordering."""
    assert _sensor_from_rows(_frame_rows(composition_offset_ticks=None)).has_bframes is False


def test_an_explicitly_zero_composition_offset_also_means_no_reordering() -> None:
    """A present-but-zero ctts box states there is no reordering, which is not the same as absent."""
    assert _sensor_from_rows(_frame_rows(composition_offset_ticks=0)).has_bframes is False


def test_a_non_zero_composition_offset_signals_reordering() -> None:
    """Composition offsets can make decode order and presentation order differ."""
    assert _sensor_from_rows(_frame_rows(composition_offset_ticks=_INDEX_FRAME_TICKS)).has_bframes is True


def test_composition_offsets_never_move_an_index_timeline() -> None:
    """Where DTS and PTS do come apart, this serves DTS and does not split the difference.

    No ``elst`` is applied, since an edit list never shifts decode timestamps, so the
    timeline is the same either way and :attr:`timestamp_source` reports which it is.
    """
    delayed = _sensor_from_rows(_frame_rows(composition_offset_ticks=_INDEX_FRAME_TICKS))

    assert delayed.has_bframes is True
    assert delayed.timestamps_ns.tolist() == _sensor_from_rows().timestamps_ns.tolist()
    assert delayed.start_ns == 0


def test_an_index_with_no_frames_is_refused_as_unusable() -> None:
    """Typed so the recipe falls back to the media; a bare ValueError would fail the stream."""
    with pytest.raises(IndexRecordUnusableError, match="no frames"):
        _sensor_from_rows([])


def test_streaming_without_a_batch_size_yields_the_whole_timeline_at_once() -> None:
    """batch_size 0 means one batch, matching CameraSensor's contract."""
    batches = list(_sensor_from_rows().stream_timestamps())

    assert len(batches) == 1
    assert batches[0].tolist() == _sensor_from_rows().timestamps_ns.tolist()


def test_streaming_splits_into_consecutive_batches_with_a_short_tail() -> None:
    """Batches partition the timeline in order, so streaming and one-shot see the same samples."""
    batches = list(_sensor_from_rows(_frame_rows(5)).stream_timestamps(batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert np.concatenate(batches).tolist() == _sensor_from_rows(_frame_rows(5)).timestamps_ns.tolist()


def test_streaming_rejects_a_negative_batch_size() -> None:
    """Mirrors CameraSensor rather than silently treating a negative size as "all"."""
    with pytest.raises(ValueError, match="batch_size must be non-negative"):
        list(_sensor_from_rows().stream_timestamps(batch_size=-1))


def test_an_index_backed_sensor_reports_the_index_and_metadata_it_was_built_from() -> None:
    """The two accessors expose the supplied metadata and frame index."""
    sensor = _sensor_from_rows()

    assert len(sensor.frame_index) == 3
    assert sensor.metadata.codec_name == "h264"


def test_sampling_a_sensor_with_no_media_raises_rather_than_yielding_nothing() -> None:
    """An index-backed sensor can answer timeline questions, but it has no packets to decode."""
    sensor = _sensor_from_rows()

    with pytest.raises(RuntimeError, match="no media to decode"):
        sensor.sample(_whole_clip(30.0, 3), policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS))


def test_an_out_of_order_timeline_is_constructible_but_not_samplable() -> None:
    """Ordering is a measurement the integrity checks report, so it cannot fail at construction.

    The checks build a sensor over whatever the recording claims and then ask it
    about itself. Refusing a descending timeline in the constructor would turn a
    finding they are there to publish into a stream that never got measured.
    Sampling such a timeline is still meaningless, so that is where it raises.
    """
    timeline = np.array([30, 20, 10], dtype=np.int64)

    sensor = _sensor_with_eof(_CLIP, timeline)

    assert sensor.timestamps_ns.tolist() == [30, 20, 10]
    assert sensor.start_ns == 30
    assert sensor.end_ns == 10

    with pytest.raises(ValueError, match="strictly sorted in ascending order"):
        sensor.sample(_whole_clip(30.0, 3), policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS))


def test_indexed_presentation_clock_reads_edited_pts() -> None:
    """Playback consumes edited PTS; the stored media PTS remains a separate, unshifted timeline."""
    elst = struct.pack(">I4sIIIihh", 28, b"elst", 0, 1, 1000, _INDEX_FRAME_TICKS, 1, 0)
    metadata = Mp4Metadata.from_row(_metadata_row(elst_bytes=elst))
    index = _frame_index(_frame_rows(), metadata)
    sensor = StreamingCameraSensor(
        None, frame_index=index, metadata=metadata, alignment_field=VideoFrameAlignmentField.EDITED_PTS_NS
    )
    assert sensor.timestamp_source is TimestampSource.CONTAINER_PTS
    np.testing.assert_array_equal(sensor.timestamps_ns, index.edited_pts_ns[index.pts_sort_order])
    assert index.pts_ns[0] == 0
    assert sensor.timestamps_ns[0] == -_indexed_ns(_INDEX_FRAME_TICKS)
    assert index.dts_ns[0] == 0


def test_indexed_sensor_owns_eof_presentation_order() -> None:
    """Recorded clocks are reordered with decoded frames and are never shifted by container edits."""
    elst = struct.pack(">I4sIIIihh", 28, b"elst", 0, 1, 1000, _INDEX_FRAME_TICKS, 1, 0)
    metadata = Mp4Metadata.from_row(_metadata_row(elst_bytes=elst))
    rows = _frame_rows()
    ranks = [2, 0, 1]
    for sample, (row, rank) in enumerate(zip(rows, ranks, strict=True)):
        row["composition_offset_ticks"] = (rank - sample + 2) * _INDEX_FRAME_TICKS
        row["eof_ns"] = _DEVICE_ORIGIN_NS + rank * _ONE_SECOND_NS
    sensor = StreamingCameraSensor(
        None,
        frame_index=_frame_index(rows, metadata),
        metadata=metadata,
        alignment_field=VideoFrameAlignmentField.EOF_NS,
    )
    assert sensor.timestamps_ns.tolist() == [_DEVICE_ORIGIN_NS + i * _ONE_SECOND_NS for i in range(3)]
    assert sensor.timestamp_source is TimestampSource.SENSOR


def test_indexed_decode_clock_is_not_shifted_by_playback_edits() -> None:
    """Integrity checks retain DTS even when the index has a shifted playback timeline."""
    elst = struct.pack(">I4sIIIihh", 28, b"elst", 0, 1, 1000, _INDEX_FRAME_TICKS, 1, 0)
    metadata = Mp4Metadata.from_row(_metadata_row(elst_bytes=elst))
    index = _frame_index(_frame_rows(), metadata)
    sensor = StreamingCameraSensor(
        None, frame_index=index, metadata=metadata, alignment_field=VideoFrameAlignmentField.DTS_NS
    )
    assert sensor.timestamp_source is TimestampSource.CONTAINER_DTS
    np.testing.assert_array_equal(sensor.timestamps_ns, index.dts_ns)
    assert sensor.timestamps_ns[0] == 0
    assert index.edited_pts_ns[0] == -_indexed_ns(_INDEX_FRAME_TICKS)


@pytest.mark.parametrize(
    ("timestamp_field", "expected_ns", "timestamp_source", "expected_frames"),
    [
        (VideoFrameAlignmentField.PTS_NS, [10, 15, 30], TimestampSource.CONTAINER_PTS, [1, 2, 0]),
        (VideoFrameAlignmentField.EDITED_PTS_NS, [5, 10, 25], TimestampSource.CONTAINER_PTS, [1, 2, 0]),
        (VideoFrameAlignmentField.EOF_NS, [100, 200, 300], TimestampSource.SENSOR, [1, 2, 0]),
        (VideoFrameAlignmentField.SOF_NS, [70, 170, 270], TimestampSource.SENSOR, [1, 2, 0]),
        (VideoFrameAlignmentField.MOE_NS, [50, 150, 250], TimestampSource.SENSOR, [1, 2, 0]),
    ],
)
def test_indexed_alignment_uses_the_selected_field_for_the_matching_frames(
    monkeypatch: pytest.MonkeyPatch,
    timestamp_field: VideoFrameAlignmentField,
    expected_ns: list[int],
    timestamp_source: TimestampSource,
    expected_frames: list[int],
) -> None:
    """Clock selection preserves frame associations across decode and presentation order."""
    metadata = Mp4Metadata.from_row(
        _metadata_row(
            time_base_den=_NS_PER_SECOND,
            elst_bytes=struct.pack(">I4sIIIihh", 28, b"elst", 0, 1, 1000, 5, 1, 0),
        )
    )
    rows = [
        {
            **row,
            "dts_ticks": dts,
            "composition_offset_ticks": offset,
            "is_keyframe": True,
            "eof_ns": eof,
            "sof_ns": eof - 30,
            "moe_ns": eof - 50,
        }
        for row, dts, offset, eof in zip(_frame_rows(), [0, 10, 20], [30, 0, -5], [300, 100, 200], strict=True)
    ]

    def decoded_frames(decoder: StreamingVideoDecoder, plan: StreamingDecodePlan) -> Iterator[tuple[int, np.ndarray]]:
        for position, count in zip(plan.frame_indices, plan.counts, strict=True):
            sample = decoder.frame_index.pts_sort_order[position]
            for _ in range(int(count)):
                yield int(position), np.full((1, 1, 3), sample, dtype=np.uint8)

    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.streaming_camera_sensor.execute_streaming_decode_plan", decoded_frames
    )
    sensor = StreamingCameraSensor(
        b"video", frame_index=_frame_index(rows, metadata), metadata=metadata, alignment_field=timestamp_field
    )
    assert sensor.timestamp_source is timestamp_source
    assert sensor.timestamps_ns.tolist() == expected_ns
    start, end = expected_ns[0], expected_ns[-1] + 1
    spec = SamplingSpec(
        grid=SamplingGrid(
            start,
            end,
            np.array(expected_ns, dtype=np.int64),
            stride_ns=end - start,
            duration_ns=end - start,
        )
    )
    served = [
        row
        for payload in sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=0))
        for row in payload.frames()
    ]
    assert [timestamp for _, timestamp, _ in served] == expected_ns
    assert [int(frame[0, 0, 0]) for _, _, frame in served] == expected_frames


def test_dts_sampling_rejects_reordered_frames_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """DTS remains readable for integrity checks without buffering pixels to restore decode order."""
    metadata = Mp4Metadata.from_row(_metadata_row())
    rows = _frame_rows(composition_offset_ticks=0)
    rows[0]["composition_offset_ticks"] = 3 * _INDEX_FRAME_TICKS
    frame_index = _frame_index(rows, metadata)
    sensor = StreamingCameraSensor(
        b"video", frame_index=frame_index, metadata=metadata, alignment_field=VideoFrameAlignmentField.DTS_NS
    )
    np.testing.assert_array_equal(sensor.timestamps_ns, frame_index.dts_ns)
    decoder = Mock(side_effect=AssertionError("unexpected decode"))
    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.streaming_camera_sensor.execute_streaming_decode_plan", decoder
    )
    with pytest.raises(ValueError, match="matching decode and presentation order"):
        sensor.sample(_whole_clip(30.0, 3), policy=NearestTimestampPolicy(max_delta_ns=0))
    decoder.assert_not_called()


@pytest.mark.parametrize(
    "alignment_field",
    [VideoFrameAlignmentField.PTS_NS, VideoFrameAlignmentField.DTS_NS, VideoFrameAlignmentField.EOF_NS],
)
def test_sampling_materializes_one_target_at_a_time(
    monkeypatch: pytest.MonkeyPatch, alignment_field: VideoFrameAlignmentField
) -> None:
    """Frames retain presentation order and are pulled only as the consumer advances."""
    count = 32
    metadata = Mp4Metadata.from_row(_metadata_row(frame_count=count, time_base_den=_NS_PER_SECOND))
    rows = _frame_rows(count)
    presentation_order = (
        list(range(count)) if alignment_field is VideoFrameAlignmentField.DTS_NS else [*range(1, count), 0]
    )
    for rank, sample in enumerate(presentation_order):
        rows[sample]["composition_offset_ticks"] = rank * 10 - rows[sample]["dts_ticks"]
        rows[sample]["eof_ns"] = 100 + rank * 10
        rows[sample]["is_keyframe"] = True
    decoded_count = 0

    def decoded_frames(_decoder: StreamingVideoDecoder, plan: StreamingDecodePlan) -> Iterator[tuple[int, np.ndarray]]:
        nonlocal decoded_count
        for rank, count in zip(plan.frame_indices, plan.counts, strict=True):
            decoded_count += 1
            for _ in range(int(count)):
                yield int(rank), np.full((1, 1, 3), presentation_order[rank], dtype=np.uint8)

    monkeypatch.setattr(
        "cosmos_curator.core.sensors.sensors.streaming_camera_sensor.execute_streaming_decode_plan", decoded_frames
    )
    sensor = StreamingCameraSensor(
        b"video", frame_index=_frame_index(rows, metadata), metadata=metadata, alignment_field=alignment_field
    )
    timestamps = sensor.timestamps_ns
    start, end = int(timestamps[0]), int(timestamps[-1]) + 1
    spec = SamplingSpec(grid=SamplingGrid(start, end, timestamps, stride_ns=end - start, duration_ns=end - start))
    payloads = sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=0))
    assert decoded_count == 0
    payload = next(payloads)
    assert decoded_count == 0
    served_count = 0
    for rank, (_align_ns, timestamp_ns, frame) in enumerate(payload.frames()):
        assert timestamp_ns == timestamps[rank]
        assert int(frame[0, 0, 0]) == presentation_order[rank]
        assert decoded_count == rank + 1
        served_count += 1
    assert served_count == count


@pytest.mark.parametrize(
    "timestamp_field",
    [VideoFrameAlignmentField.EOF_NS, VideoFrameAlignmentField.SOF_NS, VideoFrameAlignmentField.MOE_NS],
)
def test_indexed_alignment_rejects_an_absent_field(timestamp_field: VideoFrameAlignmentField) -> None:
    """Selecting an absent sensor clock cannot silently use a different field."""
    metadata = Mp4Metadata.from_row(_metadata_row())
    with pytest.raises(IndexRecordUnusableError, match=timestamp_field.value):
        _ = StreamingCameraSensor(
            None, frame_index=_frame_index(_frame_rows(), metadata), metadata=metadata, alignment_field=timestamp_field
        ).timestamps_ns


@pytest.mark.parametrize("timestamp_field", [VideoFrameAlignmentField.SOF_NS, VideoFrameAlignmentField.MOE_NS])
def test_indexed_alignment_rejects_partial_timestamps(timestamp_field: VideoFrameAlignmentField) -> None:
    """Missing timestamps cannot label decoded frames or participate in nearest-frame selection."""
    metadata = Mp4Metadata.from_row(_metadata_row())
    rows = [{**row, timestamp_field.value: value} for row, value in zip(_frame_rows(), [100, None, 300], strict=True)]
    with pytest.raises(IndexRecordUnusableError, match=rf"{timestamp_field.value}.*missing"):
        _ = StreamingCameraSensor(
            None, frame_index=_frame_index(rows, metadata), metadata=metadata, alignment_field=timestamp_field
        ).timestamps_ns


@pytest.mark.parametrize("timestamp_field", ["", "not_a_timestamp"])
def test_indexed_alignment_rejects_an_unknown_field(timestamp_field: str) -> None:
    """Invalid field names cannot silently select the default clock."""
    metadata = Mp4Metadata.from_row(_metadata_row())
    with pytest.raises(TypeError, match="VideoFrameAlignmentField"):
        StreamingCameraSensor(
            None, frame_index=_frame_index(_frame_rows(), metadata), metadata=metadata, alignment_field=timestamp_field
        )
