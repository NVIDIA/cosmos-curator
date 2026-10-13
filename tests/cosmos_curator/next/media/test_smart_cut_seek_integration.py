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

"""Seek accuracy against real ffmpeg, on a source built in the test.

The unit tests next door model how ffmpeg resolves ``-ss``. This pins that model
to the real thing: without it a change in ffmpeg's rounding would leave the unit
tests green while cuts silently started on the wrong frame.

Each source frame is a flat colour chosen so neighbours stay far apart through
yuv420p, which makes the first frame of a cut identifiable by its colour alone.
CPU only, no model weights, a few seconds.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cosmos_curator.next.media.smart_cut import _probe_index, cut_plan

pytestmark = pytest.mark.integration

_W = _H = 64
_FPS = 30
_NUM_FRAMES = 200
# Colours must survive yuv420p and stay separable; these strides keep adjacent
# frames far apart in RGB, and stay under 256 frames so no two frames collide.
_STRIDES = (53, 97, 151)


def _colour(i: int) -> tuple[int, int, int]:
    return (i * _STRIDES[0] % 256, i * _STRIDES[1] % 256, i * _STRIDES[2] % 256)


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        pytest.skip(f"{tool} not on PATH")
    return path


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Encode a constant-rate 1/30 source whose frames are individually identifiable."""
    ffmpeg = _require("ffmpeg")
    out_dir = tmp_path_factory.mktemp("smart_cut_seek")
    raw = out_dir / "raw.rgb"
    with raw.open("wb") as fh:
        for i in range(_NUM_FRAMES):
            fh.write(np.full((_H, _W, 3), _colour(i), dtype=np.uint8).tobytes())
    mp4 = out_dir / "src.mp4"
    subprocess.run(  # noqa: S603
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{_W}x{_H}", "-r", str(_FPS), "-i", str(raw),
         "-c:v", "libopenh264", "-g", "50", "-b:v", "4M", "-pix_fmt", "yuv420p",
         # Pin the timebase: 1/30 is the coarse tick that exposes the rounding.
         "-video_track_timescale", str(_FPS), str(mp4)],
        check=False, capture_output=True,
    )  # fmt: skip
    if not mp4.is_file() or mp4.stat().st_size == 0:
        pytest.skip("libopenh264 unavailable; cannot build the fixture source")
    (out_dir / "colours.json").write_text(json.dumps({i: _colour(i) for i in range(_NUM_FRAMES)}))
    return mp4


def _decoded_frame_ids(source: Path, clip: Path) -> list[int]:
    """Return which source frame each frame of *clip* is, matched on its flat colour.

    Identity rather than count: a clip that starts on the wrong frame still carries the
    requested number of frames, which is exactly what made the original defect invisible.
    """
    ffmpeg = _require("ffmpeg")
    raw = subprocess.run(  # noqa: S603
        [ffmpeg, "-v", "error", "-i", str(clip), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=False, capture_output=True,
    ).stdout  # fmt: skip
    size = _W * _H * 3
    if len(raw) < size:
        pytest.fail(f"could not decode a frame from {clip} (source {source.name})")
    ids: list[int] = []
    for off in range(0, len(raw) - size + 1, size):
        mean = np.frombuffer(raw[off : off + size], dtype=np.uint8).reshape(-1, 3).mean(axis=0)
        ids.append(
            min(range(_NUM_FRAMES), key=lambda i, m=mean: sum((a - b) ** 2 for a, b in zip(_colour(i), m, strict=True)))
        )
    return ids


def _first_frame_of(clip: Path) -> int:
    """Return which source frame the clip starts on."""
    return _decoded_frame_ids(clip, clip)[0]


@pytest.fixture(scope="module")
def transport_stream(source: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Remux the fixture to MPEG-TS, whose seek behaviour differs from MP4's.

    Seeking at a frame's own timestamp resolves to that packet on TS and starts demuxing
    without the reference keyframe, so decoding silently resumes at the next keyframe. MP4
    tolerates the same seek, which is why one container alone cannot cover the cutter.
    """
    ffmpeg = _require("ffmpeg")
    ts = tmp_path_factory.mktemp("smart_cut_ts") / "source.ts"
    subprocess.run(  # noqa: S603
        [ffmpeg, "-y", "-v", "error", "-i", str(source), "-c:v", "copy",
         "-mpegts_copyts", "1", "-muxdelay", "0", "-muxpreload", "0", str(ts)],
        check=False, capture_output=True,
    )  # fmt: skip
    if not ts.is_file() or ts.stat().st_size == 0:
        pytest.skip("could not remux the fixture to MPEG-TS")
    return ts


@pytest.fixture(params=["mp4", "ts"])
def cut_source(request: pytest.FixtureRequest, source: Path, transport_stream: Path) -> Path:
    """Run each cut test against both containers."""
    return source if request.param == "mp4" else transport_stream


# Spans chosen to exercise each cut mode against keyframes every 50: mid-GOP (full re-encode),
# exactly on a keyframe (copy), and spanning one (smart cut). The last two end on the final
# source frame, where the muxer has no following timestamp to derive the terminal sample's
# duration from -- the shape of the episode-803 case in the original report.
@pytest.mark.parametrize(
    ("start", "end"),
    [
        (31, 33),
        (50, 52),
        (1, 3),
        (45, 55),
        (101, 104),
        (118, 119),
        (_NUM_FRAMES - 3, _NUM_FRAMES - 1),
        (_NUM_FRAMES - 60, _NUM_FRAMES - 1),
    ],
)
def test_cut_starts_on_requested_source_frame(cut_source: Path, tmp_path: Path, start: int, end: int) -> None:
    """Every cut begins on the requested source frame, on both containers.

    Identity is checked against the source pixels rather than the frame count: a cut that
    starts on the wrong frame still emits the requested number of frames, which is how the
    original defect passed validation.
    """
    _require("ffmpeg")
    out = tmp_path / f"cut_{start}_{end}.mp4"
    record = cut_plan(str(cut_source), [{"startFrame": start, "endFrame": end, "output": str(out)}],
                      smart_cut=True, gpu=False)[0]  # fmt: skip
    assert record["success"], record["error"]
    assert record["written_frames"] == end - start + 1
    assert _decoded_frame_ids(cut_source, out) == list(range(start, end + 1))


def test_seek_at_frame_timestamp_alone_is_not_enough(transport_stream: Path, tmp_path: Path) -> None:
    """Guard the MPEG-TS regression: a bare target-frame seek drops the reference keyframe.

    If this stops reproducing, ffmpeg's TS seek has changed and the keyframe-anchored decode in
    ``_encode_cmd`` may no longer be load-bearing -- which is worth knowing deliberately rather
    than discovering through shifted clips.
    """
    ffmpeg = _require("ffmpeg")
    index = _probe_index(str(transport_stream))
    out = tmp_path / "naive.mp4"
    subprocess.run(  # noqa: S603
        [ffmpeg, "-y", "-v", "error", "-seek_timestamp", "1",
         "-ss", f"{index.pts[31] * index.num / index.den:.6f}", "-i", str(transport_stream),
         "-map", "0:v:0", "-an", "-frames:v", "3", "-vf", "setpts=PTS-STARTPTS",
         "-c:v", "libopenh264", "-b:v", "2M", "-pix_fmt", "yuv420p", str(out)],
        check=False, capture_output=True,
    )  # fmt: skip
    assert _decoded_frame_ids(transport_stream, out) != [31, 32, 33], (
        "a bare target-frame seek now lands correctly on MPEG-TS; re-check whether the "
        "keyframe-anchored decode is still required"
    )
