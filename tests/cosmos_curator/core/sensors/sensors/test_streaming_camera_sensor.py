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
import itertools
import pathlib
from fractions import Fraction
from typing import Any

import av
import numpy as np
import psutil
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.group import SensorGroup
from cosmos_curator.core.sensors.sensors.streaming_camera_sensor import (
    IndexRecordUnusableError,
    StreamingCameraSensor,
)
from cosmos_curator.core.sensors.types.types import TimestampSource
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
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))

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


def test_bframe_clip_needs_no_index() -> None:
    """B-frame video decodes forward-only, in presentation order, with nothing prebuilt."""
    sensor = StreamingCameraSensor(_BFRAME_CLIP, timestamps_ns=_timeline(_BFRAME_CLIP))

    pts_ns = [pts for _, pts, _ in _walk(sensor, 24.0, _ONE_MS_NS)]

    assert len(pts_ns) == _SOURCE_FRAME_COUNT
    assert pts_ns == sorted(pts_ns)
    assert len(set(pts_ns)) == _SOURCE_FRAME_COUNT


def test_one_source_frame_serves_several_reference_timestamps() -> None:
    """Sampling faster than the source rate repeats the nearest frame, without re-decoding it."""
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))

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
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))

    with pytest.raises(ValueError, match=r"max_delta_ns=1000000 exceeded"):
        for _ in _walk(sensor, 1000.0, _ONE_MS_NS, 100):
            pass


def test_peak_memory_stays_flat_over_a_long_walk() -> None:
    """Resident memory must not grow with stream length: the walk holds ~one frame.

    Walks 720 frames of 854x480 in one pass -- 0.9 GB of pixels if any of it
    were retained -- and compares peak RSS against a baseline taken once the
    decoder is warm. The threshold sits above RSS jitter rather than tight
    against it; the failure this catches is linear growth, which lands two
    orders of magnitude past it.
    """
    process = psutil.Process()
    warmup_rows = 60
    sensor = StreamingCameraSensor(_LONG_CLIP, timestamps_ns=_timeline(_LONG_CLIP))
    rows = 0
    baseline_rss = 0
    peak_rss = 0

    for _align_ns, _pts_ns, frame in _walk(sensor, 24.0, _ONE_MS_NS, _LONG_CLIP_FRAME_COUNT):
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
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))
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
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))
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
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))
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


def test_origin_pins_the_first_frame_of_a_late_starting_container(tmp_path: pathlib.Path) -> None:
    """``origin_ns`` says where the first frame lands, whatever the container calls it.

    A recording carries its capture clock beside it, not inside it, and the two
    agree on duration but not on where zero is. The caller knows when recording
    started; only the sensor knows what PTS the container gave that same frame,
    so the sensor is what reconciles them. Shifting by ``origin_ns`` instead
    would land every frame half a second late here, and exactly nowhere on a
    container that starts at zero -- which is why this needs a late one.
    """
    late = _starting_late(_CLIP, tmp_path / "late.mp4", _ONE_SECOND_NS // 2)
    with av.open(str(late)) as reader:
        stream = reader.streams.video[0]
        native_start = int(stream.start_time * stream.time_base * _NS_PER_SECOND)
    assert native_start > 0, "the remux did not preserve the shift, so this test cannot bite"

    rows = _walk(
        StreamingCameraSensor(late, timestamps_ns=_timeline(late, _DEVICE_ORIGIN_NS)),
        24.0,
        _ONE_MS_NS,
        origin_ns=_DEVICE_ORIGIN_NS,
    )

    assert rows[0][1] == _DEVICE_ORIGIN_NS
    assert rows[0][0] == _DEVICE_ORIGIN_NS


def test_origin_moves_the_timeline_without_reshaping_it() -> None:
    """Only the origin moves: the gaps between frames are the recording's own."""
    native = [
        pts for _, pts, _ in _walk(StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP)), 24.0, _ONE_MS_NS)
    ]

    shifted = [
        pts
        for _, pts, _ in _walk(
            StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP, _DEVICE_ORIGIN_NS)),
            24.0,
            _ONE_MS_NS,
            origin_ns=_DEVICE_ORIGIN_NS,
        )
    ]

    assert shifted == [pts + _DEVICE_ORIGIN_NS - native[0] for pts in native]


def _timeline(clip: pathlib.Path, origin_ns: int = 0) -> np.ndarray:
    """Return the capture times a recorder would have written beside this clip.

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
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=timeline)
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
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=timeline)
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
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))
    payloads = list(sensor.sample(_two_windows(0), policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS)))

    assert len(list(payloads[1].frames())) == 2

    with pytest.raises(RuntimeError, match="order"):
        list(payloads[0].frames())


def test_a_payload_cannot_be_resumed_once_a_later_one_has_started() -> None:
    """Interleaving is the same violation, caught at the row rather than at the start."""
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))
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
        StreamingCameraSensor(_CLIP, timestamps_ns=np.array([[1, 2], [3, 4]], dtype=np.int64))


def test_an_ascending_timeline_is_not_rejected_for_being_wide() -> None:
    """Adjacent values are compared, not differenced: a gap can exceed what int64 holds."""
    limits = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)

    assert StreamingCameraSensor(_CLIP, timestamps_ns=limits).start_ns == int(np.iinfo(np.int64).min)


def test_a_timeline_that_rounds_differently_from_the_container_still_serves() -> None:
    """Capture times and presentation timestamps are separate clocks that round apart.

    A container's time base rarely divides a nanosecond evenly: at 1/12288 this
    clip puts 80 of its 240 frames one nanosecond off where a capture clock at
    the same rate would. That is two representations of one instant, not a
    timeline describing a different recording.
    """
    step = _NS_PER_SECOND / 24.0
    timeline = np.array([round(index * step) for index in range(_SOURCE_FRAME_COUNT)], dtype=np.int64)

    rows = _walk(StreamingCameraSensor(_CLIP, timestamps_ns=timeline), 24.0, _ONE_MS_NS)

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

    rows = _walk(StreamingCameraSensor(_CLIP, timestamps_ns=timeline), 24.0, _ONE_MS_NS, origin_ns=_DEVICE_ORIGIN_NS)

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

    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=longer)

    with pytest.raises(ValueError, match="container ended"):
        _walk(sensor, 24.0, _ONE_SECOND_NS)


def test_a_container_with_more_frames_than_the_timeline_raises() -> None:
    """The reverse mismatch: a recording that outlives its own declared timeline."""
    native = _timeline(_CLIP)
    shorter = native[:-1]

    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=shorter)

    with pytest.raises(ValueError, match="holds more frames"):
        _walk(sensor, 24.0, _ONE_SECOND_NS)


def test_a_missed_exposure_is_served_by_repeating_its_neighbour() -> None:
    """A camera that skips a capture still yields a full clip, with a frame serving twice.

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
    # and again at 81's moment, so nothing was captured for 80's.
    gaps = [0] + [step if index != missed else 2 * step for index in range(1, _SOURCE_FRAME_COUNT)]
    timeline = np.cumsum(np.array(gaps, dtype=np.int64))
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=timeline)
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
    )


def _from_index(rows: list[dict[str, Any]] | None = None, **metadata_overrides: object) -> StreamingCameraSensor:
    """Build an index-backed sensor over *rows*, with no media behind it."""
    frames = _frame_rows() if rows is None else rows
    metadata = Mp4Metadata.from_row(_metadata_row(frame_count=len(frames), **metadata_overrides))
    return StreamingCameraSensor.from_index(_frame_index(frames, metadata), metadata)


def _indexed_ns(tick: int) -> int:
    return tick * _NS_PER_SECOND // _INDEX_TIME_BASE_DEN


def test_an_index_timeline_is_the_decode_timestamps_in_nanoseconds() -> None:
    """The reported timeline is dts_ticks on the track's time base, not raw ticks."""
    sensor = _from_index()

    assert sensor.timestamps_ns.dtype == np.int64
    assert sensor.timestamps_ns.tolist() == [_indexed_ns(position * _INDEX_FRAME_TICKS) for position in range(3)]
    assert sensor.start_ns == 0
    assert sensor.end_ns == _indexed_ns(2 * _INDEX_FRAME_TICKS)


def test_an_index_timeline_is_reported_as_decode_time() -> None:
    """Not merely "a container clock": decode order ascends, so an in-order finding is a tautology."""
    assert _from_index().timestamp_source is TimestampSource.CONTAINER_DTS


def test_scalar_container_facts_come_off_the_metadata_struct() -> None:
    """Codec and nominal rate are read from the row rather than inferred from the frames."""
    sensor = _from_index(codec_name="hevc", avg_frame_rate_num=30000, avg_frame_rate_den=1001)

    assert sensor.codec_name == "hevc"
    assert sensor.avg_frame_rate == Fraction(30000, 1001)


def test_a_declared_rate_of_zero_stays_zero_rather_than_raising() -> None:
    """Fraction(0) is the library's "no usable header rate" sentinel, which makes rate metrics skip."""
    assert _from_index(avg_frame_rate_num=0, avg_frame_rate_den=1).avg_frame_rate == Fraction(0)


def test_a_zero_denominator_degrades_to_the_same_sentinel() -> None:
    """A denominator the writer cannot actually emit still skips rather than raising ZeroDivisionError."""
    assert _from_index(avg_frame_rate_den=0).avg_frame_rate == Fraction(0)


def test_an_absent_composition_offset_column_means_no_reordering() -> None:
    """All-null offsets mean the ctts box is absent, so the container signals no reordering."""
    assert _from_index(_frame_rows(composition_offset_ticks=None)).has_bframes is False


def test_an_explicitly_zero_composition_offset_also_means_no_reordering() -> None:
    """A present-but-zero ctts box states there is no reordering, which is not the same as absent."""
    assert _from_index(_frame_rows(composition_offset_ticks=0)).has_bframes is False


def test_a_non_zero_composition_offset_signals_reordering() -> None:
    """A populated ctts offset is the container saying decode order and display order come apart."""
    assert _from_index(_frame_rows(composition_offset_ticks=_INDEX_FRAME_TICKS)).has_bframes is True


def test_composition_offsets_never_move_an_index_timeline() -> None:
    """Where DTS and PTS do come apart, this serves DTS and does not split the difference.

    No ``elst`` is applied, since an edit list never shifts decode timestamps, so the
    timeline is the same either way and :attr:`timestamp_source` reports which it is.
    """
    delayed = _from_index(_frame_rows(composition_offset_ticks=_INDEX_FRAME_TICKS))

    assert delayed.has_bframes is True
    assert delayed.timestamps_ns.tolist() == _from_index().timestamps_ns.tolist()
    assert delayed.start_ns == 0


def test_an_index_with_no_frames_is_refused_as_unusable() -> None:
    """Typed so the recipe falls back to the media; a bare ValueError would fail the stream."""
    with pytest.raises(IndexRecordUnusableError, match="no frames"):
        _from_index([])


def test_a_sensor_timeline_replaces_the_indexed_one_and_says_so() -> None:
    """A recorder's own clock beats the container's, and the index contributes only the scalars."""
    recorded = np.array([5_000, 15_000, 25_000], dtype=np.int64)
    metadata = Mp4Metadata.from_row(_metadata_row())

    sensor = StreamingCameraSensor.from_index(
        _frame_index(_frame_rows(), metadata),
        metadata,
        sensor_timestamps_ns=recorded,
    )

    assert sensor.timestamps_ns.tolist() == recorded.tolist()
    assert sensor.timestamp_source is TimestampSource.SENSOR
    assert sensor.codec_name == "h264"


def test_a_sensor_timeline_of_the_wrong_length_makes_the_record_unusable() -> None:
    """Matched by position, so a disagreeing count means the alignment is unknown, not partial."""
    metadata = Mp4Metadata.from_row(_metadata_row())

    with pytest.raises(IndexRecordUnusableError, match="2 observations but the frame index has 3 frames"):
        StreamingCameraSensor.from_index(
            _frame_index(_frame_rows(), metadata),
            metadata,
            sensor_timestamps_ns=np.array([5_000, 15_000], dtype=np.int64),
        )


def test_streaming_without_a_batch_size_yields_the_whole_timeline_at_once() -> None:
    """batch_size 0 means one batch, matching CameraSensor's contract."""
    batches = list(_from_index().stream_timestamps())

    assert len(batches) == 1
    assert batches[0].tolist() == _from_index().timestamps_ns.tolist()


def test_streaming_splits_into_consecutive_batches_with_a_short_tail() -> None:
    """Batches partition the timeline in order, so streaming and one-shot see the same samples."""
    batches = list(_from_index(_frame_rows(5)).stream_timestamps(batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert np.concatenate(batches).tolist() == _from_index(_frame_rows(5)).timestamps_ns.tolist()


def test_streaming_rejects_a_negative_batch_size() -> None:
    """Mirrors CameraSensor rather than silently treating a negative size as "all"."""
    with pytest.raises(ValueError, match="batch_size must be non-negative"):
        list(_from_index().stream_timestamps(batch_size=-1))


def test_an_index_backed_sensor_reports_the_index_and_metadata_it_was_built_from() -> None:
    """The two accessors are the supported way to reach what from_index was handed."""
    sensor = _from_index()

    assert len(sensor.index) == 3
    assert sensor.metadata.codec_name == "h264"


def test_a_sensor_built_over_media_has_no_index_to_report() -> None:
    """Reading an index off a sensor that never had one is a programming error, not an empty answer."""
    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=_timeline(_CLIP))

    with pytest.raises(RuntimeError, match="no frame index"):
        _ = sensor.index

    with pytest.raises(RuntimeError, match="no container metadata"):
        _ = sensor.metadata


def test_sampling_a_sensor_with_no_media_raises_rather_than_yielding_nothing() -> None:
    """An index-backed sensor can answer timeline questions, but it has no packets to decode."""
    sensor = _from_index()

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

    sensor = StreamingCameraSensor(_CLIP, timestamps_ns=timeline)

    assert sensor.timestamps_ns.tolist() == [30, 20, 10]
    assert sensor.start_ns == 30
    assert sensor.end_ns == 10

    with pytest.raises(ValueError, match="strictly sorted in ascending order"):
        sensor.sample(_whole_clip(30.0, 3), policy=NearestTimestampPolicy(max_delta_ns=_ONE_SECOND_NS))
