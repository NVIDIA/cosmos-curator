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

"""Shared fixtures for the data-integrity store tests.

These build results by running the *real* engine over a synthetic timeline rather
than hand-assembling measurements. Half of what the store tests assert is that a
value survives the trip from the kernel through Arrow and back, so a faked
measurement would fake the thing under test.

Exposed as fixtures because the test tree is not a package (pytest runs with
``--import-mode=importlib``), so one test module cannot import another's helpers.
"""

import io
import os
import pathlib
from collections.abc import Callable, Iterator
from fractions import Fraction

import av
import numpy as np
import pytest
from numpy.typing import NDArray

from cosmos_curator.core.sensors.data_integrity import identity
from cosmos_curator.core.sensors.data_integrity.engine import run_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, Thresholds
from cosmos_curator.core.sensors.data_integrity.results import (
    CheckResult,
    ResolvedConfig,
    StreamResult,
    VideoInfo,
    stream_result,
)
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.core.sensors.utils.timestamps import write_timestamps_ns
from cosmos_curator.next.recipes.data_integrity.sources import SOURCE_SENSOR_TIMESTAMPS_SUFFIX

HZ_100_PERIOD_NS = 10_000_000  # one sample every 10 ms at 100 Hz

#: A header rate of zero is the sentinel for "the container declares nothing", which
#: is what makes rate / gap / jitter never run at all.
NO_HEADER_RATE = Fraction(0, 1)


class FakeSensor:
    """Minimal stand-in satisfying the shared engine's ``IntegritySensor`` surface."""

    def __init__(
        self,
        timestamps: list[int],
        *,
        has_bframes: bool = False,
        codec_name: str = "h264",
        avg_frame_rate: Fraction = Fraction(100, 1),
        timestamp_source: TimestampSource = TimestampSource.CONTAINER_PTS,
    ) -> None:
        """Hold a synthetic timeline and the container facts the engine reads off it."""
        self._ts = np.array(timestamps, dtype=np.int64)
        self.has_bframes = has_bframes
        self.codec_name = codec_name
        self.avg_frame_rate = avg_frame_rate
        self._timestamp_source = timestamp_source

    @property
    def timestamps_ns(self) -> NDArray[np.int64]:
        """The full synthetic timeline."""
        return self._ts

    @property
    def timestamp_source(self) -> TimestampSource:
        """Which clock the synthetic timeline stands for.

        Defaults to the container's, which is what a bare list of integers honestly
        is: a timeline given no provenance. A test about sensor timestamps says so.
        """
        return self._timestamp_source

    @property
    def start_ns(self) -> int:
        """First timestamp, or zero for an empty timeline."""
        return int(self._ts[0]) if len(self._ts) else 0

    @property
    def end_ns(self) -> int:
        """Last timestamp, or zero for an empty timeline."""
        return int(self._ts[-1]) if len(self._ts) else 0

    def stream_timestamps(self, batch_size: int = 0) -> Iterator[NDArray[np.int64]]:
        """Yield the timeline in ``batch_size`` windows (``0`` = one window)."""
        step = batch_size or len(self._ts) or 1
        for start in range(0, len(self._ts), step):
            yield self._ts[start : start + step]


@pytest.fixture
def perfect() -> Callable[..., list[int]]:
    """Build ``n`` timestamps exactly one 100 Hz period apart."""

    def _make(n: int = 10) -> list[int]:
        return [i * HZ_100_PERIOD_NS for i in range(n)]

    return _make


@pytest.fixture
def drifting() -> Callable[..., list[int]]:
    """Build ``n`` timestamps at a uniformly wrong rate, ``percent`` off 100 Hz."""

    def _make(n: int = 10, *, percent: float = 1.0) -> list[int]:
        period = int(HZ_100_PERIOD_NS * (1.0 + percent / 100.0))
        return [i * period for i in range(n)]

    return _make


@pytest.fixture
def make_stream() -> Callable[..., StreamResult]:
    """Run the real engine over a synthetic timeline and package the result."""

    def _make(
        source: str,
        timestamps: list[int],
        *,
        expected_hz: float | None = 100.0,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        avg_frame_rate: Fraction = Fraction(100, 1),
        has_bframes: bool = False,
        selector_value: str = identity.DEFAULT_SELECTOR_VALUE,
        timestamp_source: TimestampSource = TimestampSource.CONTAINER_PTS,
    ) -> StreamResult:
        sensor = FakeSensor(
            timestamps,
            avg_frame_rate=avg_frame_rate,
            has_bframes=has_bframes,
            timestamp_source=timestamp_source,
        )
        metrics, info, resolved = run_metrics(sensor, expected_hz=expected_hz, thresholds=thresholds)
        return stream_result(source, metrics, info, resolved, selector_value=selector_value)

    return _make


@pytest.fixture
def run_engine() -> Callable[..., tuple[list[CheckResult], VideoInfo, ResolvedConfig]]:
    """Run the engine and return its output unpackaged, the shape the single-video CLI works in.

    :func:`make_stream` packages the same run; this returns it unpackaged, for tests
    that stand in for ``run_checks`` itself.
    """

    def _make(
        timestamps: list[int],
        *,
        expected_hz: float | None = 100.0,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        avg_frame_rate: Fraction = Fraction(100, 1),
        timestamp_source: TimestampSource = TimestampSource.CONTAINER_PTS,
    ) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
        sensor = FakeSensor(timestamps, avg_frame_rate=avg_frame_rate, timestamp_source=timestamp_source)
        return run_metrics(sensor, expected_hz=expected_hz, thresholds=thresholds)

    return _make


@pytest.fixture
def make_errored_stream() -> Callable[..., StreamResult]:
    """Build a stream that could not be opened at all -- no metrics, no measurements."""

    def _make(source: str, message: str = "moov atom not found") -> StreamResult:
        return StreamResult(
            source=source,
            codec_name=None,
            has_bframes=None,
            num_samples=None,
            start_ns=None,
            end_ns=None,
            metrics=[],
            error=message,
        )

    return _make


@pytest.fixture
def store_root(tmp_path: pathlib.Path) -> str:
    """Point at a local store root that does not exist yet, so the first write has to create it."""
    return str(tmp_path / "di-store")


# B-frame test files are NOT encoded on the fly. The ffmpeg we ship is the LGPL
# build, whose only H.264 encoder is openh264 -- and openh264 CANNOT emit
# B-frames; the encoder that can, libx264, is GPL and deliberately not bundled.
# So a ``bf=N`` encode option is silently ignored and yields a B-frame-free
# stream. Instead we read a small clip pre-encoded with libx264 + B-frames and
# checked into the repo -- decoding B-frames works in any ffmpeg build; only
# *encoding* them needs libx264.
_BFRAME_CLIP = (
    pathlib.Path(__file__).resolve().parents[3] / "pipelines" / "video" / "data" / "test_clip_10s_bframes.mp4"
)


@pytest.fixture
def h264_video() -> Callable[..., bytes]:
    """Return a factory for an H.264 MP4 with (``bframes`` > 0) or without B-frames.

    A copy of the sensor tests' fixture of the same name, which is not an ancestor
    conftest for these tests. Kept as a copy rather than hoisted to a shared ancestor
    because that conftest would import ``av`` for every collection under
    ``tests/cosmos_curator``.

    For ``bframes > 0`` it returns the checked-in libx264 B-frame clip (the bundled
    openh264 encoder can't make B-frames here -- see ``_BFRAME_CLIP``); the exact count
    is not significant, only that the stream has B-frames. For ``bframes == 0`` it
    encodes a tiny clip live (any H.264 encoder handles that).

    ``frames`` sets how long that live clip runs, at 30 fps. It is what lets a session
    hold sensors that stopped at different times, which is the only way a session-grain
    measurement has anything to measure.
    """

    def _make(*, bframes: int = 0, frames: int = 30) -> bytes:
        if bframes > 0:
            return _BFRAME_CLIP.read_bytes()
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mp4") as container:
            stream = container.add_stream("h264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
            stream.codec_context.options = {"bf": "0", "g": "30"}
            for i in range(frames):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return buffer.getvalue()

    return _make


#: Epoch-scale, so a sensor timeline cannot be mistaken for a container's own: PTS is
#: counted from the start of one file and starts at or near zero.
_SENSOR_EPOCH_NS = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z

#: One frame at 30 fps, matching what :func:`h264_video` encodes.
_FRAME_PERIOD_NS = 33_333_333


@pytest.fixture
def sensor_timeline() -> Callable[..., pathlib.Path]:
    """Return a factory writing a sensor timeline beside a video.

    The sidecar is what makes a session-grain metric run at all: without it every
    camera is measured on container PTS, the mux has already put them on one
    regenerated grid, and the two cross-sensor checks skip rather than report a rig
    that looks flawless by construction. So any test whose subject is a session
    *verdict* has to write these, and one whose subject is the skip must not.

    ``frames`` has to match the video's frame count exactly -- a timeline that does not
    fit its video is a hard error, not something to fall back from.
    """

    def _write(
        video: pathlib.Path,
        *,
        frames: int,
        start_ns: int = _SENSOR_EPOCH_NS,
        period_ns: int = _FRAME_PERIOD_NS,
    ) -> pathlib.Path:
        sidecar = pathlib.Path(f"{video}{SOURCE_SENSOR_TIMESTAMPS_SUFFIX}")
        with sidecar.open("wb") as stream:
            write_timestamps_ns(start_ns + np.arange(frames, dtype=np.int64) * period_ns, stream)
        return sidecar

    return _write


@pytest.fixture
def unreachable_video(h264_video: Callable[..., bytes]) -> Callable[[pathlib.Path], pathlib.Path]:
    """Write a real video that cannot be read, the local stand-in for an expired token.

    A mode-000 file lists like any other and then fails on open with ``PermissionError``,
    which ``session_runner`` classifies as unreachable rather than unreadable. That makes
    "the listing worked and the read did not" reachable without patching anything --
    which matters most for the pipeline tests, where the read happens in a Ray worker
    that a driver-side monkeypatch would never touch.
    """
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file regardless, so the stream would be reachable")

    def _write(path: pathlib.Path) -> pathlib.Path:
        path.write_bytes(h264_video())
        path.chmod(0o000)
        return path

    return _write
