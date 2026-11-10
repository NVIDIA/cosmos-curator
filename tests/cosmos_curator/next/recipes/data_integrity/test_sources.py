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

import attrs
import av
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.sensors.data_integrity.instruments import NAME_ORDERING
from cosmos_curator.core.sensors.data_integrity.results import CheckResult, CheckStatus
from cosmos_curator.core.sensors.mp4.box_locate import get_moov_auto
from cosmos_curator.core.sensors.mp4.moov import parse_video_track
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.data_integrity.cli_support import cancellable_reader
from cosmos_curator.next.recipes.data_integrity.session_runner import is_infrastructure_error
from cosmos_curator.next.recipes.data_integrity.sources import (
    observed_size_bytes,
    open_source,
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
        return run_checks(
            str(clip),
            expected_hz=None,
            stream_wrapper=lambda stream: cancellable_reader(stream, cancel),
            alignment_fields=_EOF_OR_PTS,
        )

    results, _info, _cfg = _check()  # type: ignore[misc]
    assert results, "the clip should evaluate normally while the event is clear"

    cancel.set()
    with pytest.raises(av.error.InvalidDataError):
        _check()


def test_a_missing_local_path_raises_at_open(tmp_path: pathlib.Path) -> None:
    """Opening eagerly moves the failure to open_source, where a missing path is exit 2."""
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

#: The preference that keeps the pre-CVC-1401 behaviour: a recorder's timeline when there
#: is one, else the container's displayed PTS. The default, ``("eof_ns",)``, refuses the
#: second half, so tests about the container clock name it explicitly.
_EOF_OR_PTS = ("eof_ns", "edited_pts_ns")


@pytest.mark.parametrize("malformed", [False, True])
@pytest.mark.parametrize("marker_suffix", [None, ".sensor_timestamps.npy", ".observations.npy"])
@pytest.mark.parametrize("field", ["pts_ns", "edited_pts_ns"])
def test_legacy_sidecars_are_never_opened_or_used(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    monkeypatch: pytest.MonkeyPatch,
    malformed: bool,  # noqa: FBT001
    marker_suffix: str | None,
    field: str,
) -> None:
    """Only an index supplies recorded clocks, regardless of nearby legacy artifacts."""
    clip = tmp_path / "front.mp4"
    clip.write_bytes(h264_video(frames=30))
    sidecar = pathlib.Path(f"{clip}.timestamps.npy")
    if malformed:
        sidecar.write_bytes(b"not a numpy file")
    else:
        np.save(sidecar, _SENSOR_EPOCH_NS + np.arange(30, dtype=np.int64) * 33_333_333)
    if marker_suffix is not None:
        pathlib.Path(f"{clip}{marker_suffix}").write_bytes(b"obsolete marker")
    opened: list[str] = []

    @contextlib.contextmanager
    def _record_open(source: str, **kwargs: object) -> object:
        opened.append(source)
        with open_source(source, **kwargs) as stream:  # type: ignore[arg-type]
            yield stream

    monkeypatch.setattr("cosmos_curator.next.recipes.data_integrity.sources.open_source", _record_open)

    _results, info, _cfg = run_checks(str(clip), expected_hz=None, alignment_fields=("eof_ns", field))

    assert info.timestamp_source is TimestampSource.CONTAINER_PTS
    assert info.start_ns == 0
    assert opened == [str(clip)]

    with pytest.raises(ValueError, match=r"EOF/SOF/MOE require input\.mp4_index_uri"):
        run_checks(str(clip), expected_hz=None)
    assert opened == [str(clip)]


def test_a_video_with_no_sensor_timeline_is_an_error_unless_container_pts_is_allowed(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """The default asks for the recorder's clock, and a stream that has none cannot answer.

    Measured on container PTS anyway, the stream would report numbers about a clock
    nobody asked for. Listing ``edited_pts_ns`` is how a run opts back in, and the bound
    proves which clock was used -- PTS is counted from the start of the file, so it
    begins at zero rather than anywhere near a wall clock.
    """
    clip = tmp_path / "front.mp4"
    clip.write_bytes(h264_video(frames=30))

    with pytest.raises(ValueError, match="no requested camera timestamps"):
        run_checks(str(clip), expected_hz=None)

    _results, info, _cfg = run_checks(str(clip), expected_hz=None, alignment_fields=_EOF_OR_PTS)
    assert info.start_ns == 0
    assert info.timestamp_source is TimestampSource.CONTAINER_PTS


def test_raw_pts_is_measured_from_the_container_without_an_index(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """``pts_ns`` is the container clock before edit lists, read from the media itself."""
    clip = tmp_path / "front.mp4"
    clip.write_bytes(h264_video(frames=30))

    _results, info, _cfg = run_checks(str(clip), expected_hz=None, alignment_fields=("pts_ns",))

    assert info.num_samples == 30
    assert info.timestamp_source is TimestampSource.CONTAINER_PTS


@pytest.mark.parametrize("field", ["eof_ns", "sof_ns", "moe_ns"])
def test_index_only_clocks_cannot_be_read_from_the_media(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    field: str,
) -> None:
    """Recorded clocks exist only in an index extension's output."""
    clip = tmp_path / "front.mp4"
    clip.write_bytes(h264_video(frames=30))

    with pytest.raises(ValueError, match=r"EOF/SOF/MOE require input\.mp4_index_uri"):
        run_checks(str(clip), expected_hz=None, alignment_fields=(field,))


@pytest.mark.parametrize(
    ("fields", "match"),
    [((), "at least one"), (("dts_ns",), "unknown camera alignment field")],
)
def test_an_unusable_preference_is_refused_before_anything_is_read(
    tmp_path: pathlib.Path, fields: tuple[str, ...], match: str
) -> None:
    """DTS is not offered: decode order always ascends, so it cannot describe timing."""
    with pytest.raises(ValueError, match=match):
        run_checks(str(tmp_path / "absent.mp4"), expected_hz=None, alignment_fields=fields)


def test_invalid_timestamps_are_not_mistaken_for_an_unreachable_source() -> None:
    """The two failure kinds stay distinct, since only one of them says the data is bad."""
    assert not is_infrastructure_error(ValueError("invalid indexed timestamps"))
    # A missing object is a fact about the dataset, so it stays a finding rather than
    # an absence of one -- which is why the 404 family is not infrastructure.
    assert not is_infrastructure_error(FileNotFoundError("front.mp4"))


def _index_for(video_bytes: bytes) -> tuple[Mp4FrameIndex, Mp4Metadata]:
    """Build the index and metadata mp4-index would store for *video_bytes*.

    Parses the real boxes rather than hand-writing a row, so these tests go through the
    same arithmetic a manifest row did.
    """

    def _fetch(offset: int, length: int) -> bytes:
        return video_bytes[offset : offset + length]

    track = parse_video_track(get_moov_auto(len(video_bytes), _fetch))
    return Mp4FrameIndex.from_video_track(track), Mp4Metadata.from_video_track(track, ftyp_bytes=None)


class _StubLookup:
    """An Mp4IndexLookup stand-in holding one prebuilt record, keyed by the path it covers.

    These tests are about which sensor ``run_checks`` chooses; the manifest read itself
    is covered in ``test_index_lookup``. ``media_size_bytes`` is recorded and not acted
    on, which is what lets these tests prove "the video was never opened" by deleting
    it -- the real guard needs the object there to size it. ``eof_ns`` stands in for
    what an index extension decodes from a recorder's own boxes; plain mp4-index output
    leaves it absent.
    """

    def __init__(self, covered: str, video_bytes: bytes, *, eof_ns: npt.NDArray[np.int64] | None = None) -> None:
        self._covered = covered
        index, metadata = _index_for(video_bytes)
        self._record = (attrs.evolve(index, eof_ns=eof_ns) if eof_ns is not None else index, metadata)
        self.asked: list[str] = []
        self.sizes: list[int | None] = []

    def index_for(
        self,
        source: str,
        *,
        media_size_bytes: int | None,
    ) -> tuple[Mp4FrameIndex, Mp4Metadata] | None:
        self.asked.append(source)
        self.sizes.append(media_size_bytes)
        if source != self._covered:
            return None
        return self._record


def _comparable(results: list[CheckResult]) -> list[tuple[object, ...]]:
    """Reduce each result to the values a store row and a report line are built from."""
    return [(r.name, r.status, r.reason, r.measurement, r.evaluation) for r in results]


def _eof_ns(frames: int) -> npt.NDArray[np.int64]:
    """Return an end-of-frame clock on the recorder's epoch, one entry per frame."""
    return _SENSOR_EPOCH_NS + np.arange(frames, dtype=np.int64) * 33_333_333


def _opened(opened: list[str]) -> Callable[[BinaryIO], BinaryIO]:
    """Return a stream wrapper that records each stream ``run_checks`` opens, by name."""

    def _wrap(stream: BinaryIO) -> BinaryIO:
        opened.append(str(getattr(stream, "name", stream)))
        return stream

    return _wrap


def test_an_indexed_stream_is_measured_on_its_eof_clock_without_opening_the_video(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """The timeline comes from the index's EOF column, proven by deleting the file after building it."""
    clip = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    clip.write_bytes(video_bytes)
    eof_ns = _eof_ns(30)
    lookup = _StubLookup(str(clip), video_bytes, eof_ns=eof_ns)
    clip.unlink()

    _results, info, _cfg = run_checks(str(clip), expected_hz=None, index_lookup=lookup)  # type: ignore[arg-type]

    assert info.num_samples == 30
    assert info.timestamp_source is TimestampSource.SENSOR
    assert info.start_ns == int(eof_ns[0])
    assert info.end_ns == int(eof_ns[-1])
    assert lookup.asked == [str(clip)]


def test_an_indexed_eof_clock_that_steps_back_is_measured_not_refused(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """A recorder clock that runs backwards is the ordering metric's finding, not a reason to skip the stream.

    ``multimodal-split`` refuses such a timeline because it cannot cut on it; this
    recipe exists to report it.
    """
    clip = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    clip.write_bytes(video_bytes)
    eof_ns = _eof_ns(30)
    eof_ns[10], eof_ns[11] = eof_ns[11], eof_ns[10]

    results, info, _cfg = run_checks(
        str(clip),
        expected_hz=None,
        index_lookup=_StubLookup(str(clip), video_bytes, eof_ns=eof_ns),  # type: ignore[arg-type]
    )

    assert info.timestamp_source is TimestampSource.SENSOR
    ordering = next(result for result in results if result.name == NAME_ORDERING)
    assert ordering.status is CheckStatus.FAIL


def test_an_indexed_stream_without_the_requested_clock_is_an_error(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """Plain mp4-index output has no EOF, and the row is current, so the video cannot supply one either."""
    clip = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    clip.write_bytes(video_bytes)
    lookup = _StubLookup(str(clip), video_bytes)
    clip.unlink()

    with pytest.raises(ValueError, match=r"no complete camera timestamp column from \['eof_ns'\]"):
        run_checks(str(clip), expected_hz=None, index_lookup=lookup)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["pts_ns", "edited_pts_ns"])
def test_an_indexed_stream_falls_through_to_a_listed_container_clock(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    field: str,
) -> None:
    """The next listed field is read from the same row, still without opening the video."""
    clip = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    clip.write_bytes(video_bytes)
    lookup = _StubLookup(str(clip), video_bytes)
    clip.unlink()

    _results, info, _cfg = run_checks(
        str(clip),
        expected_hz=None,
        index_lookup=lookup,  # type: ignore[arg-type]
        alignment_fields=("eof_ns", field),
    )

    assert info.num_samples == 30
    assert info.timestamp_source is TimestampSource.CONTAINER_PTS


def test_an_unindexed_stream_falls_back_to_opening_the_video(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """Only part of a dataset is ever indexed, so a miss has to stay an ordinary check."""
    covered = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    covered.write_bytes(video_bytes)
    other = tmp_path / "rear.mp4"
    other.write_bytes(h264_video(frames=20))

    _results, info, _cfg = run_checks(
        str(other),
        expected_hz=None,
        index_lookup=_StubLookup(str(covered), video_bytes),  # type: ignore[arg-type]
        alignment_fields=_EOF_OR_PTS,
    )

    assert info.num_samples == 20, "the fallback measured the index's video instead of this one"


def test_a_partial_index_cannot_recover_eof_from_a_sidecar(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """A manifest miss leaves EOF unavailable even if a valid legacy sidecar exists."""
    covered = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    covered.write_bytes(video_bytes)
    other = tmp_path / "rear.mp4"
    other.write_bytes(h264_video(frames=20))
    np.save(f"{other}.timestamps.npy", _eof_ns(20))

    with pytest.raises(ValueError, match=r"EOF/SOF/MOE require input\.mp4_index_uri"):
        run_checks(
            str(other),
            expected_hz=None,
            index_lookup=_StubLookup(str(covered), video_bytes),  # type: ignore[arg-type]
        )


def test_the_index_is_not_consulted_for_a_stream_it_cannot_describe(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """A manifest row is keyed by file, not by track, so it cannot answer for stream 1."""
    clip = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    clip.write_bytes(video_bytes)
    lookup = _StubLookup(str(clip), video_bytes, eof_ns=_eof_ns(30))

    with pytest.raises(IndexError):
        run_checks(
            str(clip),
            expected_hz=None,
            stream_idx=1,
            index_lookup=lookup,  # type: ignore[arg-type]
            alignment_fields=_EOF_OR_PTS,
        )

    assert lookup.asked == [], "the index answered for a stream it does not describe"


@pytest.mark.parametrize("indexed_eof", [True, False])
def test_an_indexed_stream_never_reads_the_sidecar_beside_it(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    indexed_eof: bool,  # noqa: FBT001
) -> None:
    """With a current row, the index alone decides the clock, as in ``multimodal-split``.

    The sidecar here would win under the old rule and would fail its frame-count check
    if read, so neither outcome can come from it. With EOF in the row, that clock is
    measured; without it, the stream is an error rather than a quiet fall back to the
    deprecated sidecar.
    """
    clip = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    clip.write_bytes(video_bytes)
    np.save(f"{clip}.timestamps.npy", _SENSOR_EPOCH_NS + 7 + np.arange(29, dtype=np.int64) * 33_333_333)
    eof_ns = _eof_ns(30)
    lookup = _StubLookup(str(clip), video_bytes, eof_ns=eof_ns if indexed_eof else None)
    opened: list[str] = []

    if indexed_eof:
        _results, info, _cfg = run_checks(
            str(clip),
            expected_hz=None,
            index_lookup=lookup,  # type: ignore[arg-type]
            stream_wrapper=_opened(opened),
        )
        assert info.timestamp_source is TimestampSource.SENSOR
        assert info.start_ns == int(eof_ns[0])
    else:
        with pytest.raises(ValueError, match="no complete camera timestamp column"):
            run_checks(
                str(clip),
                expected_hz=None,
                index_lookup=lookup,  # type: ignore[arg-type]
                stream_wrapper=_opened(opened),
            )
    assert opened == [], "an indexed stream opened its video or sidecar"


@pytest.mark.parametrize("bframes", [0, 2])
def test_indexed_and_walked_container_pts_measure_identically(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
    bframes: int,
) -> None:
    """The same clock read two ways must give the same timeline, bounds, and verdicts.

    The B-frame clip's ``ctts`` is one constant delay that its ``elst`` cancels exactly --
    the shape common encoders emit -- so it exercises the edit list on both paths.
    """
    clip = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30, bframes=bframes)
    clip.write_bytes(video_bytes)

    walked_results, walked_info, walked_cfg = run_checks(
        str(clip), expected_hz=None, alignment_fields=("edited_pts_ns",)
    )
    indexed_results, indexed_info, indexed_cfg = run_checks(
        str(clip),
        expected_hz=None,
        index_lookup=_StubLookup(str(clip), video_bytes),  # type: ignore[arg-type]
        alignment_fields=("edited_pts_ns",),
    )

    if bframes:
        offsets = _index_for(video_bytes)[0].composition_offset_ticks
        assert offsets is not None
        assert len(np.unique(offsets)) == 1, "the fixture is the uniform-offset case, not per-frame offsets"
        assert offsets[0] != 0
        assert indexed_info.has_bframes is True, "composition offsets are how the index sees reordering"
    assert indexed_info.timestamp_source is TimestampSource.CONTAINER_PTS
    assert indexed_info == walked_info
    assert indexed_cfg == walked_cfg
    # Compared field by field rather than as whole CheckResults: the frozen measurements
    # are attrs classes with eq=False, so two separately computed ones are never equal
    # however identical their values. These fields are what the store and the report
    # read, and they compare by value.
    assert _comparable(indexed_results) == _comparable(walked_results)


def test_the_index_is_told_how_large_the_video_is_now(
    tmp_path: pathlib.Path,
    h264_video: Callable[..., bytes],
) -> None:
    """Without this the staleness guard has nothing to compare against and would never fire."""
    clip = tmp_path / "front.mp4"
    video_bytes = h264_video(frames=30)
    clip.write_bytes(video_bytes)
    lookup = _StubLookup(str(clip), video_bytes, eof_ns=_eof_ns(30))

    run_checks(str(clip), expected_hz=None, index_lookup=lookup)  # type: ignore[arg-type]

    assert lookup.sizes == [len(video_bytes)]


def test_a_local_video_is_sized_without_being_read(tmp_path: pathlib.Path) -> None:
    """One ``os.stat``, which is the local arm of the guard's single metadata lookup."""
    clip = tmp_path / "front.mp4"
    clip.write_bytes(b"0123456789")

    assert observed_size_bytes(str(clip)) == 10


def test_a_video_that_cannot_be_sized_reports_nothing_rather_than_raising(tmp_path: pathlib.Path) -> None:
    """Being unable to decide is an answer: the caller reads the video, as it would with no index."""
    assert observed_size_bytes(str(tmp_path / "absent.mp4")) is None
