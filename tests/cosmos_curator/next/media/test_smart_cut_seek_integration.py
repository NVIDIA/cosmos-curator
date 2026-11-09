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
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cosmos_curator.next.media import smart_cut
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


@pytest.fixture(scope="module")
def vp9_source(source: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Keep the identifiable frames but use a non-H.264 codec with keyframes every 50."""
    ffmpeg = _require("ffmpeg")
    encoders = subprocess.run(  # noqa: S603
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )
    if "libvpx-vp9" not in encoders.stdout:
        pytest.skip("libvpx-vp9 unavailable; cannot build the non-H.264 fixture")
    output = tmp_path_factory.mktemp("smart_cut_vp9") / "source.mp4"
    subprocess.run(  # noqa: S603
        [ffmpeg, "-y", "-v", "error", "-i", str(source), "-an", "-c:v", "libvpx-vp9",
         "-g", "50", "-keyint_min", "50", "-lag-in-frames", "0", "-auto-alt-ref", "0",
         "-deadline", "realtime", "-cpu-used", "8", "-lossless", "1", "-threads", "1",
         "-pix_fmt", "yuv420p", "-video_track_timescale", str(_FPS), str(output)],
        capture_output=True, check=True,
    )  # fmt: skip
    return output


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


def test_validation_rejects_concealed_corruption_with_complete_frame_count(tmp_path: Path) -> None:
    """Decoder errors can preserve every frame and timestamp despite a zero probe exit code."""
    ffmpeg = _require("ffmpeg")
    ffprobe = _require("ffprobe")
    encoders = subprocess.run(  # noqa: S603
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    if not any(len(fields := line.split()) > 1 and fields[1] == "mpeg4" for line in encoders.stdout.splitlines()):
        pytest.skip("native mpeg4 encoder unavailable; cannot build the corruption fixture")

    clean = tmp_path / "clean.mp4"
    subprocess.run(  # noqa: S603
        [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10",
         "-frames:v", "10", "-c:v", "mpeg4", "-g", "1", "-q:v", "3", str(clean)],
        capture_output=True, check=True, timeout=20,
    )  # fmt: skip
    packet_probe = subprocess.run(  # noqa: S603
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets",
         "-show_entries", "packet=pos,size,flags", "-of", "json", str(clean)],
        capture_output=True, text=True, check=True, timeout=20,
    )  # fmt: skip
    packets = json.loads(packet_probe.stdout)["packets"]
    assert len(packets) == 10
    assert all("K" in packet["flags"] for packet in packets)
    packet = packets[4]
    assert int(packet["size"]) >= 64
    damaged = bytearray(clean.read_bytes())
    offset = int(packet["pos"]) + 32
    assert any(damaged[offset : offset + 32])
    # Damage the coded picture, preserving the MP4 sample table and packet boundaries.
    damaged[offset : offset + 32] = bytes(32)
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(damaged)

    probe_args = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-count_frames",
        "-show_entries",
        "stream=time_base,start_pts,duration,nb_read_frames:format=duration:frame=best_effort_timestamp",
        "-of",
        "json",
    ]
    clean_probe = subprocess.run(  # noqa: S603
        [*probe_args, str(clean)],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    corrupt_probe = subprocess.run(  # noqa: S603
        [*probe_args, str(corrupt)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert not clean_probe.stderr.strip()
    assert corrupt_probe.returncode == 0
    assert corrupt_probe.stderr.strip(), "the damaged picture must reproduce decoder errors"
    clean_metadata = json.loads(clean_probe.stdout)
    corrupt_metadata = json.loads(corrupt_probe.stdout)
    assert len(clean_metadata["frames"]) == len(corrupt_metadata["frames"]) == 10
    assert clean_metadata["streams"][0]["nb_read_frames"] == "10"
    assert corrupt_metadata["streams"][0]["nb_read_frames"] == "10"
    assert corrupt_metadata == clean_metadata, "counts, decoded PTS, and time metadata must survive corruption"

    index = _probe_index(str(clean), verify_index="always")
    assert smart_cut._validate_output(str(clean), index, 0, 10) == (10, None)
    _, error = smart_cut._validate_output(str(corrupt), index, 0, 10)
    assert error is not None, "complete counts and timing must not mask decoder errors"


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


def test_hybrid_preserves_source_timing_with_90000_timescale(tmp_path: Path) -> None:
    """The EPIC timing regression also occurs with compatible H.264 segments."""
    ffmpeg = _require("ffmpeg")
    ffprobe = _require("ffprobe")
    source = tmp_path / "50fps.mp4"
    encoded = subprocess.run(  # noqa: S603
        [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=50",
         "-frames:v", "200", "-c:v", "libopenh264", "-g", "50", "-b:v", "4M", "-threads", "1",
         "-video_track_timescale", "90000", str(source)],
        capture_output=True, check=False,
    )  # fmt: skip
    if encoded.returncode != 0:
        pytest.skip("libopenh264 unavailable; cannot build the fixture source")
    output = tmp_path / "hybrid.mp4"
    record = cut_plan(
        str(source),
        [{"startFrame": 49, "endFrame": 105, "output": str(output)}],
        smart_cut=True,
    )[0]
    assert record["success"], record
    assert record["mode"] == "smartcut", "compatible H.264 must retain the hybrid fast path"
    result = subprocess.run(  # noqa: S603
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=nb_read_frames,duration,avg_frame_rate", "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    stream = json.loads(result.stdout)["streams"][0]
    assert stream["nb_read_frames"] == "57"
    assert float(stream["duration"]) == pytest.approx(1.14, abs=0.00001)
    assert stream["avg_frame_rate"] == "50/1"


def test_non_h264_hybrid_candidate_falls_back_to_full_encode(
    vp9_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incompatible splice must become a correctly selected and timed full encode."""
    start, end = 45, 55
    index = _probe_index(str(vp9_source))
    assert index.codec_name == "vp9"
    assert index.smart_cut_safe
    assert not index.stream_copy_safe
    assert index.keyframe_at_or_before(start) == 0
    assert index.next_keyframe_at_or_after(start) == 50
    assert start < 50 <= end

    warnings = []
    monkeypatch.setattr(smart_cut.logger, "warning", lambda *args: warnings.append(args))
    output = tmp_path / "fallback.mp4"
    record = cut_plan(
        str(vp9_source),
        [{"startFrame": start, "endFrame": end, "output": str(output)}],
        smart_cut=True,
        gpu=False,
    )[0]
    assert record["success"], record
    assert record["mode"] == "encode"
    assert "vp9" in record["fallback_reason"]
    assert not warnings
    assert record["written_frames"] == end - start + 1
    assert _decoded_frame_ids(vp9_source, output) == list(range(start, end + 1))

    result = subprocess.run(  # noqa: S603
        [_require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_frames", "-count_frames",
         "-show_entries", "stream=codec_name,nb_read_frames,duration,time_base:frame=best_effort_timestamp",
         "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    metadata = json.loads(result.stdout)
    stream = metadata["streams"][0]
    assert stream["codec_name"] == "h264"
    assert int(stream["nb_read_frames"]) == end - start + 1
    assert float(stream["duration"]) == pytest.approx((end - start + 1) / _FPS, abs=0.000001)
    actual_pts = [int(frame["best_effort_timestamp"]) * Fraction(stream["time_base"]) for frame in metadata["frames"]]
    assert actual_pts == [Fraction(i, _FPS) for i in range(end - start + 1)]


@pytest.mark.parametrize("extension", ["ts", "mkv"])
@pytest.mark.parametrize("start", [0, 31, 50])
def test_copy_and_encode_to_other_containers(source: Path, tmp_path: Path, extension: str, start: int) -> None:
    """Generic CLI outputs retain their normal muxer offset and duration representation."""
    output = tmp_path / f"cut.{extension}"
    # Tiny solid-colour TS clips need enough packets for FFmpeg's format autodetection.
    count = 12
    record = cut_plan(
        str(source),
        [{"startFrame": start, "endFrame": start + count - 1, "output": str(output)}],
        smart_cut=True,
    )[0]
    assert record["success"], record
    assert _decoded_frame_ids(source, output) == list(range(start, start + count))
