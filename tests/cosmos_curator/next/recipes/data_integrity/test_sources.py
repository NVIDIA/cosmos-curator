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

"""Unit tests for turning a URI into an open sensor and running the metrics over it."""

import contextlib
import io
import pathlib
import threading
from collections.abc import Callable
from typing import BinaryIO

import av
import botocore.exceptions
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.utils.timestamps import write_timestamps_ns
from cosmos_curator.next.recipes.data_integrity.cli_support import cancellable_reader
from cosmos_curator.next.recipes.data_integrity.session_runner import is_infrastructure_error
from cosmos_curator.next.recipes.data_integrity.sources import (
    SOURCE_SENSOR_TIMESTAMPS_SUFFIX,
    open_source,
    read_sensor_timestamps,
    run_checks,
)


def _open_local(path: pathlib.Path, wrapper: object = None) -> None:
    """Read one byte from *path* through open_source, exercising the local branch."""
    with open_source(
        str(path),
        s3_profile_name=None,
        azure_profile_name="default",
        stream_wrapper=wrapper,  # type: ignore[arg-type]
    ) as stream:
        stream.read(1)


def test_a_local_path_is_opened_as_a_stream_and_wrapped(tmp_path: pathlib.Path) -> None:
    """The wrapper reaches local paths too, not just cloud URIs.

    A local path is not always a fast one: on a shared filesystem it reads like a remote
    object, so progress counting and cancellation have to apply there as well. Handing
    the sensor a Path instead would skip the wrapper entirely.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real video")
    seen: list[bytes] = []

    def _wrap(stream: BinaryIO) -> BinaryIO:
        seen.append(stream.read(3))
        stream.seek(0)
        return stream

    _open_local(clip, _wrap)

    assert seen == [b"not"], "a local path bypassed the stream wrapper, or arrived pre-read"


def test_a_local_stream_is_closed_on_exit(tmp_path: pathlib.Path) -> None:
    """open_source owns the handle it opened, so the caller cannot leak descriptors."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real video")
    captured: list[BinaryIO] = []

    with open_source(str(clip), s3_profile_name=None, azure_profile_name="default") as stream:
        captured.append(stream)
        assert not stream.closed

    assert captured[0].closed


def test_cancelling_stops_a_local_read_too(tmp_path: pathlib.Path) -> None:
    """Ctrl-C aborts a local read at the next read boundary, as it does for a cloud one."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 4096)
    cancel = threading.Event()

    with open_source(
        str(clip),
        s3_profile_name=None,
        azure_profile_name="default",
        stream_wrapper=lambda stream: cancellable_reader(stream, cancel),
    ) as stream:
        assert stream.read(16) == b"x" * 16
        cancel.set()
        assert stream.read(16) == b"", "kept reading a local file after the abort"


def test_an_abort_stops_a_real_local_check(tmp_path: pathlib.Path, h264_video: Callable[..., bytes]) -> None:
    """End to end through libav: an abort leaves a local source with nothing to decode.

    The pairing is the point. The same clip evaluates normally when the event is clear,
    so the failure below is the abort taking effect rather than a bad file, which is only
    possible because local paths now go through the wrapper instead of being handed to
    the sensor as a Path and read straight off disk.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(h264_video(bframes=0))
    cancel = threading.Event()

    def _check() -> object:
        return run_checks(str(clip), expected_hz=None, stream_wrapper=lambda stream: cancellable_reader(stream, cancel))

    results, _info, _cfg = _check()  # type: ignore[misc]
    assert results, "the clip should evaluate normally while the event is clear"

    cancel.set()
    with pytest.raises(av.error.InvalidDataError):
        _check()


def test_a_missing_local_path_raises_at_open(tmp_path: pathlib.Path) -> None:
    """Opening eagerly moves the failure to open_source, where both CLIs report exit 2."""
    with pytest.raises(FileNotFoundError):
        _open_local(tmp_path / "nope.mp4")


def test_a_cloud_source_still_takes_the_cloud_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening local paths here must not divert cloud URIs into the filesystem."""
    opened: list[str] = []

    @contextlib.contextmanager
    def _fake_open_storage_source(source: str, **_kwargs: object) -> "object":
        opened.append(source)
        yield io.BytesIO(b"cloud bytes")

    monkeypatch.setattr(
        "cosmos_curator.next.recipes.data_integrity.sources.open_storage_source", _fake_open_storage_source
    )

    with open_source("s3://bucket/key.mp4", s3_profile_name=None, azure_profile_name="default") as stream:
        assert stream.read() == b"cloud bytes"

    assert opened == ["s3://bucket/key.mp4"]


#: Epoch-scale, so a sensor timeline cannot be confused with a container's own
#: timeline: PTS is counted from the start of one file, and starts at or near zero.
_SENSOR_EPOCH_NS = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z


def _write_sensor_timeline(video: pathlib.Path, timestamps_ns: npt.NDArray[np.int64]) -> pathlib.Path:
    """Write a sensor timeline beside *video*, through the same writer the rigs' converter uses."""
    sidecar = pathlib.Path(f"{video}{SOURCE_SENSOR_TIMESTAMPS_SUFFIX}")
    with sidecar.open("wb") as stream:
        write_timestamps_ns(np.asarray(timestamps_ns, dtype=np.int64), stream)
    return sidecar


def _clip_with_timeline(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    *,
    frames: int = 30,
    entries: int | None = None,
) -> tuple[pathlib.Path, npt.NDArray[np.int64]]:
    """Write a clip plus a sensor timeline of *entries* observations (default: one per frame)."""
    clip = tmp_path / "front.mp4"
    clip.write_bytes(h264_video(frames=frames))
    count = frames if entries is None else entries
    sensor_ns = _SENSOR_EPOCH_NS + np.arange(count, dtype=np.int64) * 33_333_333
    _write_sensor_timeline(clip, sensor_ns)
    return clip, sensor_ns


def test_a_video_measured_with_no_sensor_timeline_beside_it_reports_the_containers_own(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """The ordinary case, and it has to stay silent: no sidecar is not a finding.

    Most sources have no sensor timeline, so an absent one falls back to container PTS
    without erroring. The bound proves which clock was used -- PTS is counted from the
    start of the file, so it begins at zero rather than anywhere near a wall clock.
    """
    clip = tmp_path / "front.mp4"
    clip.write_bytes(h264_video(frames=30))

    assert read_sensor_timestamps(str(clip)) is None

    _results, info, _cfg = run_checks(str(clip), expected_hz=None)
    assert info.start_ns == 0


def test_a_sensor_timeline_beside_a_video_is_what_gets_measured(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """The point of the feature, end to end through the recipe's own entry point.

    Asserted on the reported bounds rather than on the sensor, because those are what
    the session-grain metrics compare across cameras -- and comparing container PTS is
    how twelve cameras 133 ms apart came out at zero spread.
    """
    clip, sensor_ns = _clip_with_timeline(tmp_path, h264_video)

    _results, info, _cfg = run_checks(str(clip), expected_hz=None)

    assert info.num_samples == len(sensor_ns)
    assert info.start_ns == int(sensor_ns[0])
    assert info.end_ns == int(sensor_ns[-1])


def test_a_sensor_timeline_that_is_not_a_timeline_is_reported_not_fallen_back_from(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """A corrupt sidecar is a defect worth surfacing, not a reason to quietly use PTS.

    Falling back would hide the one thing the operator can act on: something wrote an
    artifact that fails its own reader.
    """
    clip = tmp_path / "front.mp4"
    clip.write_bytes(h264_video(frames=30))
    pathlib.Path(f"{clip}{SOURCE_SENSOR_TIMESTAMPS_SUFFIX}").write_bytes(b"not a npy file at all")

    with pytest.raises(ValueError, match=r"not a valid \.npy timeline"):
        run_checks(str(clip), expected_hz=None)


def test_a_sensor_timeline_that_does_not_fit_the_video_is_reported(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """A timeline of the wrong length describes some other recording.

    Entries are matched to frames by position, so a count that disagrees leaves every
    frame's sensor time in doubt rather than only the missing ones'.
    """
    clip, _sensor_ns = _clip_with_timeline(tmp_path, h264_video, frames=30, entries=29)

    with pytest.raises(ValueError, match="29 observations but the stream has 30 displayable frames"):
        run_checks(str(clip), expected_hz=None)


def test_a_timeline_is_refused_for_a_stream_it_may_not_describe(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """The sidecar names the file, so it cannot answer for one stream inside it.

    ``--stream-idx 0`` and ``--stream-idx 1`` of one container are two streams, and the
    store keys them apart -- but one timeline sits beside both. Applying it to the
    second would either put that stream on a clock measured off the first, or fail on a
    frame count that says nothing about the recording, so it is refused by name instead.
    """
    clip, _sensor_ns = _clip_with_timeline(tmp_path, h264_video)

    with pytest.raises(ValueError, match="only stream 0 is defined"):
        run_checks(str(clip), expected_hz=None, stream_idx=1)


def test_a_stream_index_is_only_refused_when_a_timeline_is_there_to_misapply(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """The refusal is about the artifact, not about the index.

    Most sources carry no timeline, and those have nothing to attribute to the wrong
    stream -- so a non-zero index goes on to be answered by the container, whatever it
    has to say about it, rather than being turned away here.
    """
    clip = tmp_path / "front.mp4"
    clip.write_bytes(h264_video(frames=30))

    with pytest.raises(IndexError):
        run_checks(str(clip), expected_hz=None, stream_idx=1)


#: Spelled out rather than imported from ``sources``, so that renaming the constant
#: there cannot quietly stop this matching what ``multimodal_split`` actually writes
#: (its ``uris.py``, ``_SENSOR_TIMESTAMPS_SUFFIX``). The name is the whole contract.
_SENSOR_TIMESTAMPS_SUFFIX = ".sensor_timestamps.npy"


def _disown_timeline(
    video: pathlib.Path, *, frames: int, marker_suffix: str = _SENSOR_TIMESTAMPS_SUFFIX
) -> pathlib.Path:
    """Write beside *video* the marker ``multimodal_split`` writes beside a clip.

    Shaped like the real one -- one entry per output frame, with the repeat a dropped
    exposure leaves behind -- so the array here is a reminder of why it cannot simply
    be read as the timeline instead: it does not strictly ascend.
    """
    marker = pathlib.Path(f"{video}{marker_suffix}")
    observed = _SENSOR_EPOCH_NS + np.arange(frames, dtype=np.int64) * 33_333_333
    observed[-1] = observed[-2]
    with marker.open("wb") as stream:
        np.save(stream, observed, allow_pickle=False)
    return marker


def test_a_resampling_grid_is_not_read_as_a_recorders_clock(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """A clip cut by ``multimodal_split`` carries a timeline no camera produced.

    That recipe resamples every camera onto one evenly spaced grid and writes the grid
    under the recorder's own filename, keeping the instants a camera observed beside it
    instead. Believed, the grid answers the session grain before it is asked: every
    camera sits on the same lattice, so spread is zero and overlap total however far
    out of sync the rig was. So the marker disowns it, and the clip is measured on
    container PTS -- which begins at zero, and is what proves which clock was used.
    """
    clip, sensor_ns = _clip_with_timeline(tmp_path, h264_video)
    _disown_timeline(clip, frames=30)

    assert read_sensor_timestamps(str(clip)) is None

    _results, info, _cfg = run_checks(str(clip), expected_hz=None)
    assert info.start_ns != int(sensor_ns[0]), "the disowned grid was measured anyway"
    assert info.start_ns == 0


def test_a_legacy_observations_marker_still_disowns_a_resampling_grid(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """Previously written multimodal clips must not make their grid look sensor-owned."""
    clip, sensor_ns = _clip_with_timeline(tmp_path, h264_video)
    _disown_timeline(clip, frames=30, marker_suffix=".observations.npy")

    assert read_sensor_timestamps(str(clip)) is None

    _results, info, _cfg = run_checks(str(clip), expected_hz=None)
    assert info.start_ns != int(sensor_ns[0]), "the legacy marker did not disown the grid"
    assert info.start_ns == 0


def test_a_marker_disowns_only_the_timeline_beside_it(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """Per stream, not per directory: twelve cameras share a clip directory.

    Both artifacts are named for one camera's media, so a marker belonging to another
    camera says nothing about this one. Keyed on the directory instead, one derived
    clip anywhere in a session would disown every real timeline in it.
    """
    clip, sensor_ns = _clip_with_timeline(tmp_path, h264_video)
    _disown_timeline(tmp_path / "rear.mp4", frames=30)

    timeline = read_sensor_timestamps(str(clip))

    assert timeline is not None, "a neighbour's marker disowned this camera's timeline"
    assert int(timeline[0]) == int(sensor_ns[0])


def test_the_stream_wrapper_reaches_the_sensor_timeline_too(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """Cancellation and byte counting have to cover the sidecar, which is fetched first.

    A wrapper that reached only the video would leave a hung timeline fetch with no way
    out: the read that blocks is the one before the one being watched.
    """
    clip, _sensor_ns = _clip_with_timeline(tmp_path, h264_video)
    wrapped: list[bytes] = []

    def _wrap(stream: BinaryIO) -> BinaryIO:
        wrapped.append(stream.read(6))
        stream.seek(0)
        return stream

    run_checks(str(clip), expected_hz=None, stream_wrapper=_wrap)

    assert len(wrapped) == 2, "the sensor timeline was opened outside the wrapper"
    assert wrapped[0] == b"\x93NUMPY", "the timeline is fetched first, so it is wrapped first"


def _wrapped_s3_error(code: str, status: int) -> OSError:
    """Build the exception ``smart_open`` raises for a failed S3 fetch.

    It replaces every ``botocore`` ``ClientError`` with a bare ``IOError``, keeping the
    original only as ``backend_error`` and as ``__cause__``. Reproduced here rather than
    mocked loosely, because that wrapper is the whole difficulty: absent, denied and
    unavailable all arrive as one exception type.
    """
    inner = botocore.exceptions.ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "GetObject",
    )
    wrapper = OSError(f"unable to access bucket: 'b' key: 'k' error: {inner}")
    wrapper.backend_error = inner  # type: ignore[attr-defined]
    wrapper.__cause__ = inner
    return wrapper


def _raise_on_open(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    """Make every cloud open fail with *exc*."""

    @contextlib.contextmanager
    def _fake_open_storage_source(_source: str, **_kwargs: object) -> "object":
        raise exc
        yield  # pragma: no cover -- unreachable, but keeps this a generator

    monkeypatch.setattr(
        "cosmos_curator.next.recipes.data_integrity.sources.open_storage_source", _fake_open_storage_source
    )


def test_a_sensor_timeline_that_is_absent_in_the_cloud_reads_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote 404 is the same "no sidecar here" as a local missing file.

    Pinned against the real wrapper shape so a ``smart_open`` change cannot silently
    turn every absent timeline into a per-stream error.
    """
    _raise_on_open(monkeypatch, _wrapped_s3_error("NoSuchKey", 404))

    assert read_sensor_timestamps("s3://bucket/front.mp4") is None


@pytest.mark.parametrize(
    ("code", "status"),
    [("AccessDenied", 403), ("InternalError", 500), ("ExpiredToken", 403)],
)
def test_an_unreachable_sensor_timeline_keeps_the_stream_unreachable(
    code: str,
    status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denial must not be read as absence, and must still count as unreachable.

    Reading it as absence would measure a whole rig against the container's metronome
    and record that as the recording's own timeline. Propagating it is only half the
    job: :func:`is_infrastructure_error` has to recognise it too, or the stream gets a
    manufactured data-quality error that supersedes an earlier healthy measurement.
    """
    exc = _wrapped_s3_error(code, status)
    _raise_on_open(monkeypatch, exc)

    with pytest.raises(OSError, match="unable to access bucket") as raised:
        read_sensor_timestamps("s3://bucket/front.mp4")

    assert is_infrastructure_error(raised.value)


def test_an_unreachable_current_marker_does_not_fall_through_to_the_legacy_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denied marker probe cannot turn a resampling grid into a trusted sensor timeline."""
    payload = io.BytesIO()
    write_timestamps_ns(np.array([_SENSOR_EPOCH_NS, _SENSOR_EPOCH_NS + 1], dtype=np.int64), payload)
    source = "s3://bucket/front.mp4"
    opened: list[str] = []
    denied = _wrapped_s3_error("AccessDenied", 403)

    @contextlib.contextmanager
    def _fake_open_storage_source(uri: str, **_kwargs: object) -> "object":
        opened.append(uri)
        if uri == f"{source}.timestamps.npy":
            yield io.BytesIO(payload.getvalue())
            return
        if uri == f"{source}.sensor_timestamps.npy":
            raise denied
        pytest.fail(f"marker probing continued after AccessDenied: {uri}")
        yield  # pragma: no cover -- unreachable, but keeps this a generator

    monkeypatch.setattr(
        "cosmos_curator.next.recipes.data_integrity.sources.open_storage_source", _fake_open_storage_source
    )

    with pytest.raises(OSError, match="unable to access bucket") as raised:
        read_sensor_timestamps(source)

    assert raised.value is denied
    assert is_infrastructure_error(raised.value)
    assert opened == [f"{source}.timestamps.npy", f"{source}.sensor_timestamps.npy"]


def test_a_bad_sensor_timeline_is_not_mistaken_for_an_unreachable_one() -> None:
    """The two failure kinds stay distinct, since only one of them says the data is bad."""
    assert not is_infrastructure_error(ValueError("stream is not a valid .npy timeline"))
    # A missing object is a fact about the dataset, so it stays a finding rather than
    # an absence of one -- which is why the 404 family is not infrastructure.
    assert not is_infrastructure_error(FileNotFoundError("front.mp4"))
