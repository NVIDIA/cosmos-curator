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

"""Tests for the ``multimodal-split`` split stage."""

import hashlib
import io
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.descriptor import FieldDescriptor
from mcap.reader import make_reader as mcap_make_reader

from cosmos_curator.core.sensors.sensors import camera_sensor
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.utils.timestamps import read_timestamps_ns, write_timestamps_ns
from cosmos_curator.next.media.spans import Span
from cosmos_curator.next.recipes.multimodal_split import split as split_module
from cosmos_curator.next.recipes.multimodal_split.config import ResolvedMultimodalSplitConfig
from cosmos_curator.next.recipes.multimodal_split.gps_streams import GpsClipWriter
from cosmos_curator.next.recipes.multimodal_split.grids import clip_duration_ns, plan_clip_spans, sampling_grid_ns
from cosmos_curator.next.recipes.multimodal_split.plan import MCAP_PLAN, PLANNING_RESULT_SCHEMA, span_planner
from cosmos_curator.next.recipes.multimodal_split.reports import plan_error_uri, sensor_error_uri
from cosmos_curator.next.recipes.multimodal_split.split import (
    SENSOR_OUTCOME_SCHEMA,
    split_aligned_session,
    split_mcap_batch,
    split_mcap_session,
)

from .conftest import DEVICE_ORIGIN_NS, gps_source_fixes, split_all

_S = 1_000_000_000
_SOURCE_FPS = 30
# A camera's timeline starts wherever the recording did, not at zero.
_EPOCH = 1_700_000_000 * _S


def test_a_timeline_that_divides_evenly_is_fully_covered() -> None:
    """Three 10 s clips tile a 30 s recording with nothing left over."""
    spans = plan_clip_spans(start_ns=_EPOCH, end_ns=_EPOCH + 30 * _S, duration_ns=10 * _S)

    assert [(s.start_ns - _EPOCH, s.end_ns - _EPOCH) for s in spans] == [
        (0, 10 * _S),
        (10 * _S, 20 * _S),
        (20 * _S, 30 * _S),
    ]


def test_a_recording_that_is_an_exact_multiple_of_the_clip_keeps_every_clip(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Fixed-length session chunks are the normal case, and none of them may lose their tail.

    60 frames at 30 fps is exactly two seconds, so a one-second clip divides it
    evenly. An off-by-one-frame usable bound shows up here as two clips instead
    of two, then as three instead of four at the next size up -- always exactly
    one clip short.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=60)
    config = make_config(root, tmp_path / "media")

    rows = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    assert len(rows) == 2
    assert not _errors(config)


def test_the_last_clip_is_served_from_real_frames(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The recovered tail clip must be genuinely servable, not a short or padded one."""
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    config = make_config(root, tmp_path / "media")

    last = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)[-1]

    assert last["start_ns"] == DEVICE_ORIGIN_NS + 2 * _S
    assert len(_observations(last)) == 30
    assert len(_decoded_means(Path(last["clip_location"]))) == 30


def test_a_short_tail_is_dropped_rather_than_kept_short() -> None:
    """The runtime generates full spans only, so 25 s at 10 s yields two clips and not three."""
    spans = plan_clip_spans(start_ns=_EPOCH, end_ns=_EPOCH + 25 * _S, duration_ns=10 * _S)

    assert len(spans) == 2
    assert spans[-1].end_ns == _EPOCH + 20 * _S


def test_spans_are_half_open_and_contiguous() -> None:
    """One clip's exclusive end is the next one's inclusive start, so no frame lands in both."""
    spans = plan_clip_spans(start_ns=_EPOCH, end_ns=_EPOCH + 30 * _S, duration_ns=10 * _S)

    assert all(left.end_ns == right.start_ns for left, right in itertools.pairwise(spans))


def test_a_recording_shorter_than_one_clip_yields_nothing() -> None:
    """A partial clip is not a clip, so a 9 s recording at 10 s produces no output."""
    assert plan_clip_spans(start_ns=_EPOCH, end_ns=_EPOCH + 9 * _S, duration_ns=10 * _S) == ()


def test_a_span_exactly_reaching_the_end_is_kept() -> None:
    """The end is exclusive, so a clip may finish on it without overrunning the recording."""
    spans = plan_clip_spans(start_ns=_EPOCH, end_ns=_EPOCH + 10 * _S, duration_ns=10 * _S)

    assert len(spans) == 1
    assert spans[0].end_ns == _EPOCH + 10 * _S


def test_an_end_before_the_start_is_rejected() -> None:
    """A camera reporting an empty or reversed range is a defect, not an empty result."""
    with pytest.raises(ValueError, match="end_ns"):
        plan_clip_spans(start_ns=_EPOCH, end_ns=_EPOCH - _S, duration_ns=10 * _S)


def _observations(clip: dict[str, Any]) -> list[int]:
    """Read which source observation served each of a clip's frames.

    Beside the clip rather than on its row: the array is one entry per frame and
    every downstream block would carry it for no reader.
    """
    media = Path(str(clip["clip_location"]))
    return [int(value) for value in np.load(media.with_name(f"{media.name}{'.observations.npy'}"))]


def _errors(config: ResolvedMultimodalSplitConfig, session_id: str = "session-000") -> list[dict[str, Any]]:
    """Read every failure a session recorded, from the reports its tasks wrote.

    Failures no longer ride the row stream, so a test asks the reports the same
    way an operator would -- and there is one per writer: planning records the
    spans no rig could serve, each camera records its own encode failures.
    """
    found: list[dict[str, Any]] = []
    for uri in (
        plan_error_uri(config.output.path, session_id),
        *(sensor_error_uri(config.output.path, session_id, key) for key in config.cameras),
    ):
        report = Path(uri)
        if report.is_file():
            found.extend(json.loads(report.read_text(encoding="utf-8"))["failures"])
    return found


def _decoded_means(path: Path) -> list[float]:
    with av.open(str(path)) as container:
        return [float(frame.to_ndarray(format="rgb24").mean()) for frame in container.decode(video=0)]


def test_a_session_yields_one_playable_clip_per_full_span(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """90 frames at 30 fps is three seconds of content, which holds three whole one-second clips.

    The last frame's timestamp is 2.9667 s, but a frame occupies the period after
    it, so the recording covers ``[0, 3)``. Tiling to the last timestamp instead
    would discard a whole clip that every one of its grid points can be served.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    config = make_config(root, tmp_path / "media")

    rows = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    clips = rows
    assert len(clips) == 3
    assert not _errors(config)
    for clip in clips:
        assert len(_observations(clip)) == 30
        assert len(_decoded_means(Path(clip["clip_location"]))) == 30


def test_one_source_frame_may_serve_several_grid_timestamps(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Provenance must be the source frame's own time, not the grid time it was chosen for.

    A 10 fps source under a 30 fps grid repeats: each source frame is the nearest
    match for about three grid points. Recording the grid timestamp instead would
    look plausible -- same length, ascending, one per row -- and would erase the
    fact that the clip holds ten distinct images rather than thirty.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=30, fps=10)
    config = make_config(root, tmp_path / "media", max_delta_s=0.06)

    clip = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)[0]

    selected = _observations(clip)
    assert len(selected) == 30
    assert len(set(selected)) < 30
    # Source frames sit 100 ms apart, so a one-second clip draws on about ten.
    assert len(set(selected)) <= 11
    # Every recorded time is a real source frame time, not a grid time. The
    # sensor is given the same origin the recipe used, because a source frame
    # time only means anything against the clock it was recorded on.
    camera_path = root / "session-000" / "camera" / "front.mp4"
    source_times = set(CameraSensor(camera_path, origin_ns=DEVICE_ORIGIN_NS).timestamps_ns.tolist())
    assert set(selected) <= source_times


def test_a_missing_camera_artifact_is_reported_rather_than_raised(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """One unusable session must not take down the run that found it."""
    root = tmp_path / "recordings"
    (root / "session-000").mkdir(parents=True)
    config = make_config(root, tmp_path / "media")

    rows = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    errors = _errors(config)
    assert not rows
    assert len(errors) == 1
    error = errors[0]
    assert error["sensor_key"] == "front"
    assert error["source_session_id"] == "session-000"
    assert error["stage"] == "open_sensor"
    assert error["category"] == "FileNotFoundError"
    assert error["message"]
    # Nothing about an unreadable file says whether reopening it would work.
    assert error["terminal_policy"] == "unclassified"
    # A session that never opened has no span, but the clip it would have been
    # is unknown too -- so neither is invented.
    assert error["start_ns"] is None
    assert error["clip_id"] is None


def test_a_tolerance_no_frame_can_meet_fails_the_clip_and_writes_nothing(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip that cannot be served is an error with no media, not a clip with substituted frames.

    The source runs at 10 fps and the grid at 30 fps, so two out of every three
    grid points sit about 33 ms from the nearest frame. A 1 ms tolerance rejects
    them rather than accepting a frame a third of a second away.

    Refused from the timeline rather than partway through decoding: which
    observation serves a reference is decided by capture times, so a recording
    that cannot meet the tolerance is known before a container is opened.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=30, fps=10)
    media = tmp_path / "media"
    config = make_config(root, media, max_delta_s=0.001)

    rows = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    assert not rows
    errors = _errors(config)
    assert list(media.rglob("*.mp4")) == []
    first = errors[0]
    assert first["stage"] == "coverage"
    assert first["category"] == "AlignmentError"
    # The recording cannot serve this grid and never will, so a retry driver must
    # not be told to try again.
    assert first["terminal_policy"] == "fail_clip"
    # A failed clip names the object it would have written, so a quarantine
    # reader can tell whether a stale payload from an earlier run is sitting
    # at that key.
    assert first["clip_id"]
    assert first["start_ns"] == DEVICE_ORIGIN_NS
    assert first["exclusive_end_ns"] == DEVICE_ORIGIN_NS + _S


def test_a_descriptor_carries_what_publication_will_need(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The later Lance ticket reads these rows, so the facts it needs are recorded now."""
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    config = make_config(root, tmp_path / "media")

    clip = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)[0]

    assert clip["source_session_id"] == "session-000"
    assert clip["sensor_key"] == "front"
    assert clip["exclusive_end_ns"] - clip["start_ns"] == 1_000_000_000
    # One observation per frame of the clip's own grid: a second of 30 fps.
    assert len(_observations(clip)) == 30
    assert clip["clip_id"]


def test_a_clip_is_written_under_its_session_and_clip_in_the_output_path(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The layout is what a later publication step joins against, so it is part of the contract."""
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    media = tmp_path / "media"
    config = make_config(root, media)

    clip = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)[0]

    expected = media / "sessions" / "session-000" / clip["clip_id"] / "front.mp4"
    assert clip["clip_location"] == str(expected)
    assert expected.is_file()


def test_the_configured_output_rate_reaches_the_written_clip(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A rate that never leaves the config is a setting the operator cannot actually use.

    At 60 fps the same one-second clip holds twice the rows and the file plays at
    twice the rate. Asserting the descriptor alone would pass against a hardcoded
    constant, so the written media is checked too.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90, fps=60)
    config = make_config(root, tmp_path / "media", output_fps=60, duration_s=1.0)

    clip = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)[0]

    assert len(_observations(clip)) == 60
    path = Path(clip["clip_location"])
    assert len(_decoded_means(path)) == 60
    with av.open(str(path)) as container:
        assert container.streams.video[0].average_rate == 60


@pytest.mark.parametrize(
    ("duration_s", "output_fps", "expected_ns"),
    [(4.1, 30, 4_100_000_000), (2.05, 60, 2_050_000_000), (8.2, 25, 8_200_000_000), (10.0, 30, 10_000_000_000)],
)
def test_a_clip_duration_converts_to_nanoseconds_exactly(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    duration_s: float,
    output_fps: int,
    expected_ns: int,
) -> None:
    """Seconds are float and nanoseconds are integer, and the conversion between them truncates.

    ``4.1 * 1e9`` is ``4099999999.9999995`` in binary floating point, so a float
    conversion loses a nanosecond. Every span boundary would then start early,
    and the error grows along the recording rather than cancelling.
    """
    config = make_config(
        tmp_path / "recordings",
        tmp_path / "media",
        duration_s=duration_s,
        output_fps=output_fps,
    )

    assert clip_duration_ns(config) == expected_ns


def test_a_fractional_duration_lands_on_exact_nanosecond_boundaries(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Clip geometry is integer nanoseconds, and float seconds do not convert exactly.

    0.7 s is 0.7000000000000000622 in binary floating point. Multiplying by 1e9
    and truncating gives 699999999, so every span would start a nanosecond early
    and drift further along the recording.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    config = make_config(root, tmp_path / "media", duration_s=0.7, output_fps=30)

    clips = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    assert [clip["start_ns"] for clip in clips] == [
        DEVICE_ORIGIN_NS + offset for offset in (0, 700_000_000, 1_400_000_000, 2_100_000_000)
    ]
    assert all(len(_observations(clip)) == 21 for clip in clips)


def test_a_short_encode_fails_the_clip_instead_of_publishing_it(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clip holding fewer frames than its grid is not a clip, however it came about.

    Nothing reachable produces this today; it guards the seam between selection
    and encoding, so a future decoder that quietly returns short cannot publish a
    clip whose descriptor claims rows the media does not have.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    media = tmp_path / "media"
    config = make_config(root, media)
    monkeypatch.setattr(split_module, "encode_frames_to_path", lambda *_args, **_kwargs: 1)

    rows = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    assert not rows
    assert all(error["category"] == "ValueError" for error in _errors(config))
    assert "whose grid has 30" in _errors(config)[0]["message"]
    assert list(media.rglob("*.mp4")) == []


def test_a_recording_shorter_than_one_clip_yields_no_rows(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Nothing to emit is a quiet result, not an error."""
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=10)
    config = make_config(root, tmp_path / "media")

    rows = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    assert rows == []
    # Quiet means quiet: a recording too short to fill a clip is a legitimate
    # empty result, and reporting it as a failure would bury the real ones.
    assert _errors(config) == []


def test_a_camera_is_opened_as_a_stream_rather_than_a_path(tmp_path: Path, make_session: Callable[..., Path]) -> None:
    """The sensor library is dropping its ``Path`` arm, so nothing here may rely on it.

    Opening every backend the same way is also what keeps a scheme check out of
    this recipe: ``smart_open`` returns a seekable stream for a local path, a
    ``file://`` URI, and an ``s3://`` URI alike.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")

    with split_module.open_sensor_source(str(root / "session-000" / "camera" / "front.mp4")) as source:
        assert isinstance(source, io.BufferedIOBase)
        assert source.readable()
        assert source.seekable()


def test_an_unreadable_camera_emits_one_failure_outcome_per_planned_span(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An early open failure cannot make every episode for that camera disappear from accounting."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    config = make_config(root, tmp_path / "media")
    plan = span_planner(
        {"source_session_id": "session-000", "session_uri": str(session)},
        config=config,
    )[0]
    monkeypatch.setattr(
        split_module,
        "open_sensor_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("camera unavailable")),
    )

    outcomes = split_module.split_session(plan, config=config)

    assert len(outcomes) == len(plan["spans"])
    assert {outcome["status"] for outcome in outcomes} == {"failure"}
    assert all(outcome["expected_row_count"] == 30 for outcome in outcomes)


def test_splitting_builds_no_video_index(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indexing is the one eager pass a forward-only walk exists to avoid.

    On B-frame video an index costs a full demux of the source before a single
    frame is decoded. The only thing it was wanted for here was the first and
    last timestamp, and the timeline artifact carries both. Nothing else fails
    if an index creeps back in: it would be correct, and slow, and silent.
    """
    calls: list[int] = []
    original = camera_sensor.make_index_and_metadata
    monkeypatch.setattr(
        camera_sensor,
        "make_index_and_metadata",
        lambda *args, **kwargs: (calls.append(1), original(*args, **kwargs))[1],
    )
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}

    clips = split_all(item, config=make_config(root, tmp_path / "media"))

    assert clips
    assert calls == []


def test_a_clip_is_stamped_in_the_recorder_clock_not_the_container_clock(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A span is capture time, which is the only thing that joins a camera to the other sensors.

    Every container's presentation timestamps start at zero, so spans taken from
    them describe a position within one file and nothing else: two sessions
    recorded a day apart claim the same instants, and none of them line up with
    an imu or gps reading. The timeline artifact carries the recorder's own
    clock, and that is what a clip is published under.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    expected = [DEVICE_ORIGIN_NS, DEVICE_ORIGIN_NS + _S, DEVICE_ORIGIN_NS + 2 * _S]

    clips = split_all(item, config=make_config(root, tmp_path / "media"))

    assert [clip["start_ns"] for clip in clips] == expected
    assert [clip["exclusive_end_ns"] for clip in clips] == [start + _S for start in expected]


def test_a_clip_is_written_with_a_timeline_beside_it(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip needs the same thing its source needed: when its frames happened.

    An MP4 starts at zero however it was cut, so a clip cannot say where it sits
    on the recorder's clock, and a later stage would be back to inferring it. The
    sidecar is the same convention the recording itself arrives with, so the next
    stage reads a clip exactly as this stage read a session.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    config = make_config(root, tmp_path / "media")

    clips = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    for clip in clips:
        media = Path(clip["clip_location"])
        beside = media.with_name(f"{media.name}.timestamps.npy")
        assert beside.exists()
        with beside.open("rb") as stream:
            timeline = read_timestamps_ns(stream)
        assert len(timeline) == len(_observations(clip))
        assert int(timeline[0]) == clip["start_ns"]
        assert int(timeline[-1]) < clip["exclusive_end_ns"]


def _two_camera_session(root: Path, make_session: Callable[..., Path], *, left_origin_ns: int) -> dict[str, Any]:
    """Return a session with two cameras on one clock, the left one starting when told."""
    make_session(root, "session-000", frames=90, uri_pattern="camera/front.mp4")
    make_session(root, "session-000", frames=90, uri_pattern="camera/left.mp4", origin_ns=left_origin_ns)
    return {"source_session_id": "session-000", "session_uri": str(root / "session-000")}


_TWO_CAMERAS = {"front": "camera/front.mp4", "left": "camera/left.mp4"}


def test_every_camera_is_written_under_the_same_clip_id(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip is one moment across every view, so the views share an address.

    The clip id is the moment; the camera key is a file inside it. That is what
    lets a reader ask for "everything recorded at this instant" without knowing
    which cameras a run was configured with.
    """
    root = tmp_path / "recordings"
    item = _two_camera_session(root, make_session, left_origin_ns=DEVICE_ORIGIN_NS)
    config = make_config(root, tmp_path / "media", cameras=_TWO_CAMERAS)

    clips = split_all(item, config=config)

    by_id: dict[str, set[str]] = {}
    for clip in clips:
        by_id.setdefault(clip["clip_id"], set()).add(clip["sensor_key"])
    assert all(cameras == {"front", "left"} for cameras in by_id.values())
    assert len(by_id) == 3


def test_spans_start_where_every_camera_can_serve_them(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The common window, not the first camera's: a reference no view covers is not a moment.

    Cameras on one rig start within a few frames of each other. Planning from one
    of them would put the first references before another view began recording.
    """
    root = tmp_path / "recordings"
    late_ns = DEVICE_ORIGIN_NS + 400_000_000
    item = _two_camera_session(root, make_session, left_origin_ns=late_ns)
    config = make_config(root, tmp_path / "media", cameras=_TWO_CAMERAS)

    clips = split_all(item, config=config)

    assert {clip["start_ns"] for clip in clips} == {late_ns, late_ns + _S}


def test_a_moment_a_camera_cannot_serve_is_refused_before_anything_is_written(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """One view short is not a multiview clip, and no view is written for it.

    Both cameras record the whole window, so the common bounds do not exclude
    anything -- but the left one runs at 10 fps against a 30 fps grid, and cannot
    put an observation within a millisecond of most references. The moment is
    refused for the front camera too, which could have served it.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90, uri_pattern="camera/front.mp4")
    make_session(root, "session-000", frames=30, fps=10, uri_pattern="camera/left.mp4")
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    media = tmp_path / "media"
    config = make_config(root, media, cameras=_TWO_CAMERAS, max_delta_s=0.001)

    rows = split_all(item, config=config)

    assert not rows
    assert list(media.rglob("*.mp4")) == []
    assert {row["stage"] for row in _errors(config)} == {"coverage"}
    assert {row["sensor_key"] for row in _errors(config)} == {"left"}


def test_a_wildcard_finds_a_camera_without_naming_its_recorder(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A rig spreads its cameras over recorders, and the config should not have to know which.

    Twelve cameras across five recorders is a layout that belongs to the rig, not
    to a run. Naming it in the config means editing twelve paths whenever a
    recorder is rewired, and getting one wrong is a view silently missing.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90, uri_pattern="recorder-07/camera_front.mp4")
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    config = make_config(root, tmp_path / "media", uri_pattern="*/camera_front.mp4")

    clips = split_all(item, config=config)

    assert len(clips) == 3
    assert all(clip["clip_location"].endswith("front.mp4") for clip in clips)


def test_a_wildcard_matching_nothing_is_a_missing_camera(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Nothing matched means this recording does not hold the camera that was asked for."""
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90, uri_pattern="recorder-07/camera_front.mp4")
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    config = make_config(root, tmp_path / "media", uri_pattern="*/camera_rear.mp4")

    split_all(item, config=config)
    errors = _errors(config)

    assert errors[0]["category"] == "FileNotFoundError"
    assert "matches" in errors[0]["message"] or "no artifact" in errors[0]["message"]


def test_a_wildcard_matching_several_names_no_camera_at_all(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Two recorders holding the same camera name is a different failure from none holding it.

    Picking one would drop a view from every clip in the run and say nothing.
    Which one it picked would depend on listing order, so the same config would
    produce different datasets on different days.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90, uri_pattern="recorder-07/camera_front.mp4")
    make_session(root, "session-000", frames=90, uri_pattern="recorder-08/camera_front.mp4")
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    config = make_config(root, tmp_path / "media", uri_pattern="*/camera_front.mp4")

    split_all(item, config=config)
    errors = _errors(config)

    assert errors[0]["category"] == "ValueError"
    assert "matches 2 artifacts" in errors[0]["message"]


def test_a_clip_records_which_observation_served_each_frame(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The grid says when a clip's frames are; this says when the camera actually saw them.

    A clip's timeline is the grid it was resampled onto, evenly spaced by
    construction. The recording it came from was not: cameras jitter and
    occasionally miss an exposure. Without this the difference is gone the moment
    the clip is written, and a later stage splitting that clip again would build
    on an idealized clock with no way back to the original instants.

    Non-decreasing rather than ascending, because one observation can serve
    several frames -- which is exactly what a missed exposure looks like.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    config = make_config(root, tmp_path / "media")

    clips = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    for clip in clips:
        media = Path(clip["clip_location"])
        observed = np.load(media.with_name(f"{media.name}.observations.npy"))
        assert observed.dtype == np.int64
        assert len(observed) == len(_observations(clip))
        assert all(left <= right for left, right in itertools.pairwise(observed))


def test_a_clip_names_the_session_and_span_it_came_from(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip on disk should not need a table to say where it came from.

    Its own timeline says where its frames sit, but nothing in the directory says
    which session was cut or which part of it, so a reader holding only the
    output tree cannot place the clip in the recording it came from.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    config = make_config(root, tmp_path / "media", cameras=_TWO_CAMERAS)
    make_session(root, "session-000", frames=90, uri_pattern="camera/left.mp4")

    clips = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    clip = clips[0]
    manifest = json.loads((Path(clip["clip_location"]).parent / "clip.json").read_text())
    assert manifest["parent_id"] == "session-000"
    assert manifest["clip_id"] == clip["clip_id"]
    assert manifest["start_ns"] == clip["start_ns"]
    assert manifest["exclusive_end_ns"] == clip["exclusive_end_ns"]
    assert manifest["sampling_fps"] == 30
    assert sorted(manifest["planned_cameras"]) == ["front", "left"]


def test_a_rerun_over_the_same_output_rewrites_identical_bytes(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The ordering holds a clip together within a generation; this is why there is only one.

    Sidecars are written before the media, so a failure between them leaves a
    clip invisible rather than unreadable. Running again over a finished clip
    replaces the sidecars before the MP4, which would mix two generations if a
    rerun could produce different bytes -- old media beside new observations,
    and no error, because the MP4 exists and the clip looks published.

    It cannot, because the same clip id means the same content by construction:
    the id covers the source, the span, the rates, the encoder, the camera map
    and the tolerance. So the second pass overwrites every artifact with what
    was already there. Mixing generations needs the source to change under a
    stable id, which breaks the identity contract before it reaches the writes.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    media = tmp_path / "media"
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    config = make_config(root, media)

    first = split_all(item, config=config)
    assert first
    before = {path: path.read_bytes() for path in sorted(media.rglob("*")) if path.is_file()}

    second = split_all(item, config=config)

    assert [clip["clip_id"] for clip in second] == [clip["clip_id"] for clip in first]
    after = {path: path.read_bytes() for path in sorted(media.rglob("*")) if path.is_file()}
    assert sorted(after) == sorted(before), "a rerun added or removed an artifact"
    assert after == before, "a rerun rewrote an artifact with different bytes"


def test_a_clip_is_never_published_without_its_sidecars(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The media is written last, so a failure cannot leave a clip that cannot place itself.

    A consumer walking the output looks for the MP4. Publishing it before its
    timeline and observations would leave a window where a clip exists and
    cannot say when its frames happened -- and a failure between the writes
    would leave that permanently, under an id a rerun addresses again.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    media = tmp_path / "media"
    config = make_config(root, media)

    real_write = split_module.write_media

    def fail_on_observations(location: str, data: bytes, *, storage_profile: str) -> None:
        if location.endswith(".observations.npy"):
            msg = "storage went away"
            raise OSError(msg)
        real_write(location, data, storage_profile=storage_profile)

    monkeypatch.setattr(split_module, "write_media", fail_on_observations)

    clips = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    assert clips == []
    assert list(media.rglob("*.mp4")) == []
    assert _errors(config)[0]["stage"] == "sample_and_encode"


def test_a_run_that_went_well_writes_nothing_under_errors(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A directory named errors should be empty when there were none.

    Both stages write only on failure. A receipt per healthy session would be a
    write per session before any media exists -- three per session at twelve
    sensors -- and it would defend against a session vanishing unnoticed, which
    Ray Data does not permit: a task that raises kills the dataset.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000")
    media = tmp_path / "media"
    config = make_config(root, media)

    clips = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    assert clips
    assert not Path(plan_error_uri(str(media), "session-000")).exists()
    assert not Path(sensor_error_uri(str(media), "session-000", "front")).exists()
    assert not (media / "errors").exists()


def test_one_sensors_failure_leaves_the_other_sensors_clips_and_report_alone(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run is mixed far more often than it is all-healthy or all-broken.

    Twelve cameras and one bad encoder is the ordinary bad run, and two things
    have to hold that neither uniform case can show. The healthy sensor keeps
    its clips, because a sensor is its own row and its own task. And the report
    is named for the sensor that failed, so the one that did not stays absent
    from the errors tree rather than being buried under a shared file: a single
    split_session.json would let the last writer decide whose failures survive.
    """
    root = tmp_path / "recordings"
    item = _two_camera_session(root, make_session, left_origin_ns=DEVICE_ORIGIN_NS)
    media = tmp_path / "media"
    config = make_config(root, media, cameras=_TWO_CAMERAS)

    real_write = split_module.write_media

    def fail_the_left_camera(location: str, data: bytes, *, storage_profile: str) -> None:
        if "left.mp4" in location:
            msg = "storage went away"
            raise OSError(msg)
        real_write(location, data, storage_profile=storage_profile)

    monkeypatch.setattr(split_module, "write_media", fail_the_left_camera)

    clips = split_all(item, config=config)

    assert {clip["sensor_key"] for clip in clips} == {"front"}
    assert [path.name for path in sorted(media.rglob("*.mp4"))] == ["front.mp4"] * len(clips)
    assert not Path(sensor_error_uri(str(media), "session-000", "front")).exists()
    recorded = Path(sensor_error_uri(str(media), "session-000", "left"))
    assert recorded.is_file()
    written = json.loads(recorded.read_text(encoding="utf-8"))
    assert {failure["sensor_key"] for failure in written["failures"]} == {"left"}

    # The manifest is written during planning, so it says what the clip was meant
    # to hold. The media beside it says what it holds. This is the case where the
    # two differ, and why the field is not called ``cameras``.
    manifest = json.loads((Path(clips[0]["clip_location"]).parent / "clip.json").read_text(encoding="utf-8"))
    assert manifest["planned_cameras"] == ["front", "left"]
    assert sorted(path.stem for path in Path(clips[0]["clip_location"]).parent.glob("*.mp4")) == ["front"]


def test_a_timeline_that_outruns_its_media_reports_every_span_it_cost(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Planning trusts the timeline, so only the walk can find it wrong.

    The sensor is handed a timeline rather than asked to verify one -- checking
    means walking the file, which is the cost it exists to avoid. A sidecar
    describing more observations than its container holds is therefore found
    partway through, and every span after that point is unproducible too. Each
    is recorded, because a clip that never appears and never fails is one nobody
    goes looking for: the run is simply short, and nothing says why.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    config = make_config(root, tmp_path / "media", duration_s=1.0)
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}

    # Twice the observations the container can serve, so planning proposes spans
    # the media runs out partway through.
    timeline_path = next((root / "session-000").rglob("*.timestamps.npy"))
    written = np.load(timeline_path)
    step = int(written[1] - written[0])
    stretched = np.arange(int(written[0]), int(written[0]) + 180 * step, step, dtype=np.int64)
    with timeline_path.open("wb") as handle:
        write_timestamps_ns(stretched, handle)

    planned = {
        str(entry["clip_id"])
        for row in span_planner(item, config=config, storage_profile="default")
        for entry in row["spans"]
    }
    clips = split_all(item, config=config)

    recorded = _errors(config)
    produced = {clip["clip_id"] for clip in clips}
    reported = {entry["clip_id"] for entry in recorded}

    assert produced, "the case needs a span the media could serve"
    assert reported, "and one it could not"
    assert produced | reported == planned, "a planned span neither appeared nor was reported"
    assert not produced & reported
    assert "stream_ended" in {entry["stage"] for entry in recorded}


def test_a_refused_span_says_which_stage_failed_and_why(
    tmp_path: Path,
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Breadcrumbs are only worth writing if they name the cause.

    The file is named for the stage that wrote it, so a directory of them says
    which step of a run went wrong without opening one; inside, each entry
    carries the stage and the exception, because "produced nothing" is not a
    reason.
    """
    root = tmp_path / "recordings"
    (root / "session-000").mkdir(parents=True)
    media = tmp_path / "media"
    config = make_config(root, media)

    clips = split_all({"source_session_id": "session-000", "session_uri": str(root / "session-000")}, config=config)

    recorded = Path(plan_error_uri(str(media), "session-000"))
    assert clips == []
    assert recorded.name == "span_planner.json"
    written = json.loads(recorded.read_text(encoding="utf-8"))
    assert written["stage"] == "span_planner"
    assert written["failures"][0]["category"]
    assert written["failures"][0]["message"]


def test_a_failed_clip_does_not_shift_the_frames_of_the_clips_after_it(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming pass is shared across spans and cannot seek back.

    A clip that fails partway leaves the cursor inside its own span: the encoder
    stopped reading, but the frames it read are gone. Without realigning, the
    next span starts mid-clip and every clip after it holds frames from the
    wrong moment -- written with no error, because nothing compares a clip's
    frames against the span it claims.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    reference = split_all(item, config=make_config(root, tmp_path / "clean", duration_s=1.0))

    real_encode = split_module.encode_frames_to_path
    calls = {"n": 0}

    def fail_the_first_clip_partway(frames: object, path: Path, **kwargs: object) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            for index, _frame in enumerate(frames):  # type: ignore[call-overload]
                if index == 10:
                    msg = "encoder gave up"
                    raise RuntimeError(msg)
            return 0
        return real_encode(frames, path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(split_module, "encode_frames_to_path", fail_the_first_clip_partway)
    config = make_config(root, tmp_path / "damaged", duration_s=1.0)

    written = split_all(item, config=config)

    # The first clip is lost; the ones after it must be the frames they would
    # have held had it never failed.
    assert [clip["clip_id"] for clip in written] == [clip["clip_id"] for clip in reference][1:]
    for clip, expected in zip(written, reference[1:], strict=True):
        assert _observations(clip) == _observations(expected)


def test_a_failure_that_ends_the_pass_early_reports_the_spans_after_it(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other way the pass ends, with spans still to come.

    A clip can fail partway and leave too few frames to catch up on, so
    realigning ends the pass rather than the feed running dry inside a clip.
    Both paths strand every span after them, but only this one leaves the failed
    span already recorded -- so it must report the ones after it and not itself
    again.

    The last-span test reaches this branch too, and cannot show it works: there
    is nothing after the last span, so the list it builds is empty either way.
    """
    root = tmp_path / "recordings"
    # Five spans planned from the timeline, seventy frames of media: the third
    # span is where the container runs out, and two spans follow it.
    make_session(root, "session-000", frames=70)
    config = make_config(root, tmp_path / "media", duration_s=1.0)
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}

    timeline_path = next((root / "session-000").rglob("*.timestamps.npy"))
    written = np.load(timeline_path)
    step = int(written[1] - written[0])
    with timeline_path.open("wb") as handle:
        write_timestamps_ns(np.arange(int(written[0]), int(written[0]) + 150 * step, step, dtype=np.int64), handle)

    real_encode = split_module.encode_frames_to_path
    calls = {"n": 0}

    def fail_the_third_clip_immediately(frames: object, path: Path, **kwargs: object) -> int:
        calls["n"] += 1
        if calls["n"] == 3:
            for index, _frame in enumerate(frames):  # type: ignore[call-overload]
                if index == 2:
                    msg = "encoder gave up"
                    raise RuntimeError(msg)
            return 0
        return real_encode(frames, path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(split_module, "encode_frames_to_path", fail_the_third_clip_immediately)

    planned = [
        str(entry["clip_id"])
        for row in span_planner(item, config=config, storage_profile="default")
        for entry in row["spans"]
    ]
    clips = split_all(item, config=config)

    recorded = _errors(config)
    stages = {entry["clip_id"]: entry["stage"] for entry in recorded}
    produced = {clip["clip_id"] for clip in clips}

    assert len(planned) > 3, "the case needs spans after the one that fails"
    assert produced == set(planned[:2])
    # The clip that failed is reported once, as itself, not as a casualty.
    assert stages[planned[2]] == "sample_and_encode"
    # Everything after it is stranded, and each says so exactly once.
    assert [stages[clip_id] for clip_id in planned[3:]] == ["stream_ended"] * len(planned[3:])
    assert len(recorded) == len(planned) - len(produced)


def test_a_failure_on_the_last_span_is_still_reported(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realigning must not swallow the outcome it was realigning after.

    A clip that fails near the tail leaves the cursor short of the boundary, and
    there may be no frames left to catch up on. Realigning then ends the pass --
    which is correct -- but the failure that caused it still happened and still
    has to be reported.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    config = make_config(root, tmp_path / "media", duration_s=1.0)

    real_encode = split_module.encode_frames_to_path
    calls = {"n": 0}

    def fail_the_last_clip_partway(frames: object, path: Path, **kwargs: object) -> int:
        calls["n"] += 1
        if calls["n"] == 3:
            for index, _frame in enumerate(frames):  # type: ignore[call-overload]
                if index == 10:
                    msg = "encoder gave up"
                    raise RuntimeError(msg)
            return 0
        return real_encode(frames, path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(split_module, "encode_frames_to_path", fail_the_last_clip_partway)

    written = split_all(item, config=config)

    assert len(written) == 2, "the two clips before the failure should still be written"
    recorded = _errors(config)
    assert [entry["stage"] for entry in recorded] == ["sample_and_encode"]


def test_realigning_past_the_end_of_the_stream_is_refused() -> None:
    """A span the stream cannot finish leaves too few frames to catch up on.

    This is the case that makes the ordering matter: realigning has to be able
    to fail, so the outcome it was realigning after must already be recorded.
    """
    frames = ((index, index, np.zeros((2, 2, 3), dtype=np.uint8)) for index in range(15))
    feed = split_module._FrameFeed(frames)
    selected: list[int] = []

    taking = feed.take(30, selected)
    for index, _frame in enumerate(taking):
        if index == 9:
            break
    taking.close()

    assert feed.taken == 10
    with pytest.raises(split_module._StreamEndedError):
        feed.realign(30)


def test_two_failures_in_a_row_leave_the_next_clip_where_it_belongs(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realigning has to hold across consecutive failures, not just one.

    Each failure leaves the cursor at its own offset inside its own span, so the
    catch-up is cumulative: a second failure realigns from wherever the first
    one left it. Get that wrong and the survivor after them holds frames from
    somewhere between the two.
    """
    root = tmp_path / "recordings"
    make_session(root, "session-000", frames=90)
    item = {"source_session_id": "session-000", "session_uri": str(root / "session-000")}
    reference = split_all(item, config=make_config(root, tmp_path / "clean", duration_s=1.0))

    real_encode = split_module.encode_frames_to_path
    calls = {"n": 0}
    # Different offsets, so the two catch-ups differ and a fixed-size drop would
    # not save it.
    give_up_at = {1: 5, 2: 22}

    def fail_the_first_two(frames: object, path: Path, **kwargs: object) -> int:
        calls["n"] += 1
        stop = give_up_at.get(calls["n"])
        if stop is not None:
            for index, _frame in enumerate(frames):  # type: ignore[call-overload]
                if index == stop:
                    msg = "encoder gave up"
                    raise RuntimeError(msg)
            return 0
        return real_encode(frames, path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(split_module, "encode_frames_to_path", fail_the_first_two)
    config = make_config(root, tmp_path / "damaged", duration_s=1.0)

    written = split_all(item, config=config)

    assert [clip["clip_id"] for clip in written] == [reference[2]["clip_id"]]
    assert _observations(written[0]) == _observations(reference[2])
    assert len(_errors(config)) == 2


def _ego_clips(
    session: Path,
    config: ResolvedMultimodalSplitConfig,
) -> list[dict[str, object]]:
    """Plan one session and cut its MCAP sensor, as the pipeline does."""
    plans = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)
    ego_plan = next(row for row in plans if row["result_kind"] == MCAP_PLAN)
    return split_mcap_session(ego_plan, config=config)


def test_an_mcap_clip_is_written_per_span_and_catalogued(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Every planned span becomes one MCAP beside the clip's video, with its digest."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    outcomes = _ego_clips(session, config)

    assert [outcome["status"] for outcome in outcomes] == ["success"] * 3
    assert all(outcome["sensor_type"] == "egotrajectory" for outcome in outcomes)
    for outcome in outcomes:
        media = Path(str(outcome["media_location"]))
        assert media.name == "ego.mcap"
        assert media.parent.name == outcome["split_clip_id"]
        written = media.read_bytes()
        assert outcome["media_size_bytes"] == len(written)
        assert outcome["media_sha256"] == hashlib.sha256(written).hexdigest()
        with media.open("rb") as stream:
            decoded = list(mcap_make_reader(stream).iter_messages())
        assert decoded
    # Copied rather than resampled, so there is no grid and no per-row account of
    # which sample served which row -- the two artifacts a camera clip carries.
    assert all(outcome["timeline_location"] is None for outcome in outcomes)
    assert all(outcome["observations_location"] is None for outcome in outcomes)


def test_an_interior_mcap_clip_carries_a_pose_at_or_before_its_start(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip can answer for its own first instant, which a strict half-open cut cannot.

    Aligning grid row zero asks for a pose at exactly ``start``, and nearest
    selection needs a sample at or before it. The session's first clip has none
    to carry -- nothing precedes the recording -- so it is the interior clips
    that show the bracket doing its job.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    outcomes = _ego_clips(session, config)

    answerable = []
    carried = []
    for outcome in outcomes:
        with Path(str(outcome["media_location"])).open("rb") as stream:
            times = [message.log_time for _s, _c, message in mcap_make_reader(stream).iter_messages()]
        start_ns = int(outcome["start_ns"])
        answerable.append(sum(1 for log_time in times if log_time <= start_ns))
        carried.append(sum(1 for log_time in times if log_time < start_ns))
    # The first clip answers for its own first instant from its own messages,
    # because this trajectory begins exactly where the cameras do. The later two
    # only manage it because the cut carried one message in from before the span.
    assert carried == [0, 1, 1]
    assert answerable == [1, 2, 2]


def test_an_mcap_clip_holds_exactly_its_span_plus_the_bracket(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The cut is half-open on the bounds the cameras were cut on, plus one message.

    A second of a 100 Hz trajectory is 100 messages. The session's first clip has
    exactly those; the later ones carry the bracket as well, which is the message
    the clip before it ended on.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    outcomes = _ego_clips(session, config)

    counts = []
    for outcome in outcomes:
        with Path(str(outcome["media_location"])).open("rb") as stream:
            counts.append(len(list(mcap_make_reader(stream).iter_messages())))
    assert counts == [100, 101, 101]


def test_an_unreadable_mcap_fails_every_span_without_ending_the_session(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A trajectory that cannot be opened reports one failure per planned span.

    Per span, not one for the sensor: the outcome stream is terminal, and a span
    absent from it is indistinguishable from a worker that died holding it.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    ego = make_ego_mcap(session)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})
    plans = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)
    ego_plan = next(row for row in plans if row["result_kind"] == MCAP_PLAN)
    ego.write_bytes(b"not an mcap")

    outcomes = split_mcap_session(ego_plan, config=config)

    assert [outcome["status"] for outcome in outcomes] == ["failure"] * 3
    assert all(outcome["media_location"] is None for outcome in outcomes)


def test_a_clip_spanning_a_trajectory_gap_is_refused_with_its_reason(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A clip holding no poses of its own fails, and the report says why.

    Planning cannot see a dropout -- it reads the topic's first and last time and
    nothing between -- so this span is a legitimate plan output. The clip carries
    only the pose borrowed from before it, which cannot motion-compensate
    anything, so it is refused rather than published looking whole.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session, gap=(100, 200))
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    outcomes = _ego_clips(session, config)

    assert [outcome["status"] for outcome in outcomes] == ["success", "failure", "success"]
    report = json.loads(Path(sensor_error_uri(config.output.path, "session-000", "ego")).read_bytes())
    assert len(report["failures"]) == 1
    assert report["failures"][0]["stage"] == "cut_mcap"
    assert "no messages between" in report["failures"][0]["message"]


def test_the_bracket_is_the_nearest_earlier_pose_on_an_unchunked_recording(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A recorder that wrote no chunk indexes still yields the pose just before the span.

    The MCAP reader silently drops ``reverse`` when it cannot seek, so a reverse
    lookup there returns the session's *earliest* message. On a three-hour
    recording that is a pose hours stale, presented as the sample immediately
    before the clip -- and nothing downstream could tell.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session, chunked=False)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    outcomes = _ego_clips(session, config)

    brackets = []
    for outcome in outcomes:
        with Path(str(outcome["media_location"])).open("rb") as stream:
            times = sorted(message.log_time for _s, _c, message in mcap_make_reader(stream).iter_messages())
        earlier = [log_time for log_time in times if log_time < int(outcome["start_ns"])]
        brackets.append(earlier[-1] - int(outcome["start_ns"]) if earlier else None)
    # 10ms before each span, one sample of a 100Hz trajectory -- not the session
    # origin, which is where the unguarded reverse lookup lands.
    assert brackets == [None, -10_000_000, -10_000_000]


def test_a_multi_topic_mcap_must_name_the_topic_to_cut(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Cutting one of several topics without being told which would drop the rest."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session, extra_topic="/other")
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    results = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)

    # Refused while planning, which reads the topic's bounds -- the ambiguity is
    # caught before any clip directory exists rather than once per span.
    assert results == []
    report = json.loads(Path(plan_error_uri(config.output.path, "session-000")).read_bytes())
    assert report["failures"][0]["sensor_key"] == "ego"
    assert "must be named in config" in report["failures"][0]["message"]


def test_a_named_topic_is_cut_from_a_multi_topic_mcap(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Naming the topic cuts that one and leaves its neighbour out of the clip."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session, extra_topic="/other")
    config = make_config(
        root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"}, topic="/egotrajectory"
    )

    outcomes = _ego_clips(session, config)

    assert [outcome["status"] for outcome in outcomes] == ["success"] * 3
    with Path(str(outcomes[0]["media_location"])).open("rb") as stream:
        topics = {channel.topic for _s, channel, _m in mcap_make_reader(stream).iter_messages()}
    assert topics == {"/egotrajectory"}


def test_a_cut_clip_carries_the_bytes_and_schema_it_arrived_with(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Every field of every message survives the cut, and so does the schema.

    This is the whole contract: records are copied, never decoded, so a clip
    opens with the code that already reads the session. Checking only timestamps
    and counts would pass just as happily against a cutter that rewrote every
    payload, which is exactly what makes a clip silently useless -- a consumer
    reading it gets well-timed garbage.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    ego = make_ego_mcap(session)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})

    with ego.open("rb") as stream:
        source = {
            message.log_time: (
                schema.name,
                schema.encoding,
                schema.data,
                channel.message_encoding,
                message.data,
                message.publish_time,
                message.sequence,
            )
            for schema, channel, message in mcap_make_reader(stream).iter_messages()
        }
    outcomes = _ego_clips(session, config)

    seen = 0
    for outcome in outcomes:
        with Path(str(outcome["media_location"])).open("rb") as stream:
            for schema, channel, message in mcap_make_reader(stream).iter_messages():
                assert (
                    schema.name,
                    schema.encoding,
                    schema.data,
                    channel.message_encoding,
                    message.data,
                    message.publish_time,
                    message.sequence,
                ) == source[message.log_time]
                seen += 1
    assert seen == 302  # 300 of their own across three clips, plus two brackets


def test_an_mcap_block_round_trips_through_the_terminal_schema(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The batch stage casts sensor rows to the same schema camera rows use.

    Which is where the null sidecar descriptors are actually exercised: a schema
    that still demanded them would pass every direct-call test and fail only once
    Ray asked for a block.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session)
    config = make_config(root, tmp_path / "media", mcap_sensors={"ego": "ego/egotrajectory.mcap"})
    plans = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)
    block = pa.Table.from_pylist(
        [row for row in plans if row["result_kind"] == MCAP_PLAN], schema=PLANNING_RESULT_SCHEMA
    )

    outcomes = split_mcap_batch(block, config=config)

    assert outcomes.schema == SENSOR_OUTCOME_SCHEMA
    rows = outcomes.to_pylist()
    assert [row["status"] for row in rows] == ["success"] * 3
    assert all(row["timeline_location"] is None for row in rows)
    assert all(row["sensor_type"] == "egotrajectory" for row in rows)


def _gps_clips(
    session: Path,
    config: ResolvedMultimodalSplitConfig,
) -> list[dict[str, object]]:
    """Plan one session and align its GPS sensor, as the pipeline does."""
    plans = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)
    gps_plan = next(row for row in plans if row["sensor_key"] == "gps")
    return split_aligned_session(gps_plan, config=config)


def _rows(location: object) -> list[tuple[int, int, float, float, float, dict[str, str]]]:
    """Read a clip's GPS back as (log_time, payload_ns, lat, lon, alt, provenance)."""
    path = Path(str(location))
    with path.open("rb") as stream:
        reader = mcap_make_reader(stream)
        schema = next(iter(reader.get_summary().schemas.values()))
        pool = descriptor_pool.DescriptorPool()
        for file_descriptor in descriptor_pb2.FileDescriptorSet.FromString(schema.data).file:
            pool.Add(file_descriptor)
        message_cls = message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))
        rows = []
        for _s, _c, message in reader.iter_messages(log_time_order=True):
            fix = message_cls.FromString(message.data)
            rows.append(
                (
                    message.log_time,
                    fix.timestamp.seconds * 1_000_000_000 + fix.timestamp.nanos,
                    fix.latitude,
                    fix.longitude,
                    fix.altitude,
                    {kv.key: kv.value for kv in fix.metadata},
                )
            )
        return rows


@pytest.mark.parametrize("sampling", ["nearest_neighbor", "interpolate"])
def test_gps_lands_on_the_same_grid_the_cameras_do(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
    sampling: str,
) -> None:
    """Row i of the position is the instant of frame i, which is the point of one timeline.

    A sensor-specific rate would hand a consumer back the correspondence problem
    the episode exists to remove, so the grid is the camera's regardless of what
    rate the fixes arrived at. The payload carries the same instant as the
    envelope, matching the source, where the two agree.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session)
    config = make_config(
        root, tmp_path / "media", gps_sensors={"gps": "gps/gps.mcap"}, sampling=sampling, duration_s=1.0
    )

    outcomes = _gps_clips(session, config)

    assert [outcome["status"] for outcome in outcomes] == ["success"] * 3
    for outcome in outcomes:
        rows = _rows(outcome["media_location"])
        grid = sampling_grid_ns(
            Span(start_ns=int(outcome["start_ns"]), end_ns=int(outcome["exclusive_end_ns"])), config
        )
        assert [log_time for log_time, *_rest in rows] == [int(value) for value in grid]
        assert [payload for _t, payload, *_rest in rows] == [int(value) for value in grid]
        assert len(rows) == outcome["expected_row_count"]


def test_nearest_selection_copies_the_fix_it_names(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """Every row is a recorded position, equal to the fix its provenance points at.

    Checking every component against the source is what distinguishes a correct
    row from one holding the right latitude in the wrong field, or the nearest
    fix's time beside a different fix's position.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session)
    source = gps_source_fixes()
    config = make_config(
        root, tmp_path / "media", gps_sensors={"gps": "gps/gps.mcap"}, sampling="nearest_neighbor", duration_s=1.0
    )

    rows = _rows(_gps_clips(session, config)[0]["media_location"])

    assert len(rows) == 30
    for log_time, _payload, lat, lon, alt, meta in rows:
        assert meta["cosmos_curator.sampling"] == "nearest_neighbor"
        named = int(meta["cosmos_curator.source_timestamp_ns"])
        assert (lat, lon, alt) == source[named]
        # The fix it names is the nearest one, not merely one that exists.
        assert named == min(source, key=lambda fix_ns: (abs(fix_ns - log_time), fix_ns))
    # A 1 Hz source against a 30 Hz grid repeats: two fixes bracket a one-second
    # clip, so the rows hold two distinct positions rather than thirty.
    assert len({row[2] for row in rows}) == 2


def test_interpolation_places_each_row_between_the_fixes_it_names(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A derived row sits on the line between its two bracketing fixes, in every component.

    Asserting only that rows ascend would accept any monotonic sequence, so this
    pins each row against the fixes its own provenance names.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session)
    source = gps_source_fixes()
    config = make_config(
        root, tmp_path / "media", gps_sensors={"gps": "gps/gps.mcap"}, sampling="interpolate", duration_s=1.0
    )

    rows = _rows(_gps_clips(session, config)[0]["media_location"])

    assert len({row[2] for row in rows}) == len(rows)
    for log_time, _payload, lat, lon, alt, meta in rows:
        assert meta["cosmos_curator.sampling"] == "interpolate"
        lower, upper = (int(value) for value in meta["cosmos_curator.source_bracket_timestamps_ns"].split(","))
        assert lower <= log_time <= upper
        # The brackets are the adjacent pair, not merely a pair that straddles.
        assert upper - lower <= 1_000_000_000
        alpha = 0.0 if upper == lower else (log_time - lower) / (upper - lower)
        # Interpolation runs in ECEF, so a derived point sits on the chord between
        # the two fixes rather than on the geodetic-linear line. Over a second of
        # driving that is about a micrometre of altitude -- far below these
        # tolerances, which are in turn far below the per-fix step a swapped
        # component or a wrong alpha would show.
        for index, value, tolerance in ((0, lat, 1e-7), (1, lon, 1e-7), (2, alt, 1e-3)):
            expected = source[lower][index] + alpha * (source[upper][index] - source[lower][index])
            assert value == pytest.approx(expected, abs=tolerance)


def test_a_stale_fix_beyond_the_tolerance_refuses_only_its_own_clip(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A dropped fix costs the clips it left unsupported, and no others.

    The bound exists to separate a fix that arrived late from one that never
    arrived. Without a test that trips it, a bound wired to the wrong policy or
    ignored entirely would look identical to one doing its job.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session, dropped=frozenset({3, 4}))
    config = make_config(
        root,
        tmp_path / "media",
        gps_sensors={"gps": "gps/gps.mcap"},
        sampling="nearest_neighbor",
        duration_s=1.0,
    )

    outcomes = _gps_clips(session, config)

    statuses = [outcome["status"] for outcome in outcomes]
    assert "failure" in statuses
    assert "success" in statuses
    report = json.loads(Path(sensor_error_uri(config.output.path, "session-000", "gps")).read_bytes())
    assert report["failures"][0]["stage"] == "align_sensor"
    assert "max_delta" in report["failures"][0]["message"]


def test_an_unreadable_gps_source_fails_every_span_and_says_so(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A sensor that cannot be opened reports one failure per planned span.

    Per span, not one for the sensor: the outcome stream is terminal, and a span
    absent from it cannot be told apart from a worker that died holding it.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    gps = make_gps_mcap(session)
    config = make_config(
        root, tmp_path / "media", gps_sensors={"gps": "gps/gps.mcap"}, sampling="interpolate", duration_s=1.0
    )
    plans = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)
    gps_plan = next(row for row in plans if row["sensor_key"] == "gps")
    gps.write_bytes(b"not an mcap")

    outcomes = split_aligned_session(gps_plan, config=config)

    assert [outcome["status"] for outcome in outcomes] == ["failure"] * 3
    assert all(outcome["media_location"] is None for outcome in outcomes)
    report = json.loads(Path(sensor_error_uri(config.output.path, "session-000", "gps")).read_bytes())
    assert report["failures"][0]["stage"] == "open_sensor"


def test_a_source_message_without_the_fields_gps_rows_need_is_refused_once(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
) -> None:
    """Naming the missing field beats an AttributeError raised once per clip.

    The writer fills named position fields, so a producer whose message lacks one
    cannot be written at all. Finding that out when the sensor opens reports it
    once, against the sensor, rather than once per clip as an unclassified fault.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")

    missing = make_gps_mcap(session, uri_pattern="a.mcap").read_bytes()
    with pytest.raises(ValueError, match=r"carries no heading field"):
        GpsClipWriter(
            missing,
            "/gps",
            required_fields=(("heading", FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.LABEL_OPTIONAL, ()),),
        )

    # Spelled right, shaped wrong. A check that only looked at names would wave
    # these through and fail identically once per clip, which is the whole thing
    # it exists to prevent.
    wrong_timestamp = make_gps_mcap(session, uri_pattern="b.mcap", timestamp_as_int=True).read_bytes()
    with pytest.raises(ValueError, match=r"has timestamp as type"):
        GpsClipWriter(wrong_timestamp, "/gps")

    singular = make_gps_mcap(session, uri_pattern="c.mcap", singular_metadata=True).read_bytes()
    with pytest.raises(ValueError, match=r"has metadata where a repeated one is needed"):
        GpsClipWriter(singular, "/gps")


def test_a_block_routes_each_sensor_to_the_cut_its_kind_needs(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_ego_mcap: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """One stage, two cuts: the trajectory is copied and the position is placed on the grid.

    Both arrive as the same plan kind because both come from an MCAP and both are
    cheap next to video. Only how the clip is written differs, and a block holding
    one of each is where that dispatch is actually exercised.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_ego_mcap(session)
    make_gps_mcap(session)
    config = make_config(
        root,
        tmp_path / "media",
        mcap_sensors={"ego": "ego/egotrajectory.mcap"},
        gps_sensors={"gps": "gps/gps.mcap"},
        sampling="interpolate",
        duration_s=1.0,
    )
    plans = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)
    block = pa.Table.from_pylist(
        [row for row in plans if row["result_kind"] == MCAP_PLAN], schema=PLANNING_RESULT_SCHEMA
    )

    outcomes = split_mcap_batch(block, config=config).to_pylist()

    assert len(outcomes) == 6
    assert all(row["status"] == "success" for row in outcomes)
    by_sensor = {row["sensor_key"]: row for row in outcomes}
    assert set(by_sensor) == {"ego", "gps"}
    # The trajectory keeps its recorded rate; the position is one row per grid point.
    ego_rows = len(_rows_log_times(by_sensor["ego"]["media_location"]))
    gps_rows = len(_rows_log_times(by_sensor["gps"]["media_location"]))
    assert gps_rows == by_sensor["gps"]["expected_row_count"]
    assert ego_rows != gps_rows


def _rows_log_times(location: object) -> list[int]:
    """Return every message time in a clip, whatever its schema."""
    with Path(str(location)).open("rb") as stream:
        return [message.log_time for _s, _c, message in mcap_make_reader(stream).iter_messages()]


def test_a_tightened_tolerance_refuses_rows_the_default_would_serve(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The configured bound is the one applied, not the default it replaced.

    On an undropped 1 Hz source the furthest a grid row sits from the nearest fix
    is half a second, which the default tolerates and 400 ms does not. So a run
    that ignored the configured value -- or read the interpolation bound instead
    -- would publish these clips rather than refuse them, and no assertion about
    healthy data could tell.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session)
    config = make_config(
        root,
        tmp_path / "media",
        gps_sensors={"gps": "gps/gps.mcap"},
        sampling="nearest_neighbor",
        max_delta_ms=400,
        duration_s=1.0,
    )

    outcomes = _gps_clips(session, config)

    assert [outcome["status"] for outcome in outcomes] == ["failure"] * 3
    report = json.loads(Path(sensor_error_uri(config.output.path, "session-000", "gps")).read_bytes())
    assert "max_delta" in report["failures"][0]["message"]

    # And the same data under the default bound is served, so the refusal above
    # is the tolerance biting rather than the fixture being unservable.
    relaxed = make_config(
        root, tmp_path / "media2", gps_sensors={"gps": "gps/gps.mcap"}, sampling="nearest_neighbor", duration_s=1.0
    )
    assert [outcome["status"] for outcome in _gps_clips(session, relaxed)] == ["success"] * 3


def test_a_source_whose_envelope_and_payload_disagree_is_refused(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """A converter that transforms one clock and not the other cannot produce a clip.

    Planning bounds the window by log_time and alignment runs on the payload, so
    when they disagree the window and the samples describe different clocks.
    Every clip then fails with something that reads like missing data. Refusing
    the sensor once, naming the gap, is what makes the converter the suspect.
    """
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session, payload_skew_ns=200_000_000)
    config = make_config(
        root, tmp_path / "media", gps_sensors={"gps": "gps/gps.mcap"}, sampling="interpolate", duration_s=1.0
    )
    plans = span_planner({"source_session_id": "session-000", "session_uri": str(session)}, config=config)
    gps_plan = next(row for row in plans if row["sensor_key"] == "gps")

    outcomes = split_aligned_session(gps_plan, config=config)

    assert [outcome["status"] for outcome in outcomes] == ["failure"] * len(outcomes)
    report = json.loads(Path(sensor_error_uri(config.output.path, "session-000", "gps")).read_bytes())
    assert report["failures"][0]["stage"] == "open_sensor"
    assert "a gap of 200000000 ns" in report["failures"][0]["message"]


def test_a_source_whose_clocks_agree_is_not_refused(
    tmp_path: Path,
    make_session: Callable[..., Path],
    make_gps_mcap: Callable[..., Path],
    make_config: Callable[..., ResolvedMultimodalSplitConfig],
) -> None:
    """The check must not refuse the healthy case it is meant to let through."""
    root = tmp_path / "recordings"
    session = make_session(root, "session-000")
    make_gps_mcap(session)
    config = make_config(
        root, tmp_path / "media", gps_sensors={"gps": "gps/gps.mcap"}, sampling="interpolate", duration_s=1.0
    )

    assert [outcome["status"] for outcome in _gps_clips(session, config)] == ["success"] * 3
