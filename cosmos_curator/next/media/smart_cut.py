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

"""Frame-exact, low-RAM video cutter (per-cut PTS-seek, with a stream-copy fast path).

Motivation
----------
Cutting clips from a chunk mp4 must be frame-exact on CFR **and** VFR and any codec
(H.264, AV1, ...), while keeping RAM low enough to pack many workers per node.

* ``decord`` decoding is frame-exact but its per-batch memory is never released, so
  RSS creeps into the tens-of-GB range on large / high-res chunks, and it cannot
  decode AV1 at all.
* Piping whole raw frames through Python to a fan-out of encoders (a single decode
  pass) is VFR-safe but pushes gigabytes of rgb24 per chunk, making it slower and
  heavier than a plain per-cut ffmpeg on real (AV1) inputs.

This module reads each frame's presentation timestamp (PTS) from the container's
**packets** (``ffprobe -show_packets`` -- no decode), then cuts each range with an
independent ffmpeg that fast-seeks to the exact frame and either **stream-copies**
(all-intra / GOP-1 sources: no decode or re-encode -- fastest, lowest RAM, lossless)
or re-encodes by exact PTS (VFR-safe). Nothing is piped through Python, so peak RSS is
one ffmpeg (tens of MB) and cuts are frame-exact regardless of frame rate. Frame
positions are never derived from a constant frame rate.

Re-encode settings mirror ``run_extract_subtask_clips._encode_frames_to_mp4_streaming``
for output parity.

Plan format (JSON)::

    {"source": "chunk.mp4", "cuts": [{"startFrame": 0, "endFrame": 239, "output": "cut01.mp4"}, ...]}

``startFrame``/``endFrame`` are **inclusive** frame indices.
"""

import argparse
import bisect
import functools
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO, Any

from loguru import logger

_DEFAULT_BITRATE = "2M"
_PROBE_TIMEOUT_S = 600
_CUT_TIMEOUT_S = 600
_PROBE_RETRIES = 2

# Layouts where ONE packet is guaranteed to be exactly ONE displayed frame, so the packet-derived
# index is the frame index and the (decoding, therefore expensive) cardinality check can be skipped.
# ISOBMFF stores one coded picture per sample (ISO/IEC 14496-15) and progressive H.264/HEVC has no
# field pairing or temporal-unit packing. Everything else -- AV1/VP9 (temporal units, superframes,
# show_existing_frame), MPEG-TS, Matroska, interlaced, unknown -- must be verified by decoding.
_ONE_PACKET_PER_FRAME_FORMATS = frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"})
_ONE_PACKET_PER_FRAME_CODECS = frozenset({"h264", "hevc"})
_INTERLACED_FIELD_ORDERS = frozenset({"tt", "bb", "tb", "bt"})
_VERIFY_INDEX_MODES = frozenset({"auto", "always", "never"})
# Seconds of stream decoded by the "auto" cardinality check. Packets-per-frame is a systematic
# property of the encoder+muxer, so a bounded prefix exposes a violation immediately: on a split-
# temporal-unit AV1 file the prefix reads 149 packets vs 100 frames. Measured on the real AV1
# sources this costs ~0.5s versus ~95-115s to decode the whole file (~150x cheaper).
_PREFIX_VERIFY_SECONDS = 10


@functools.lru_cache(maxsize=1)
def _nvenc_available() -> bool:
    """Return True if the local ffmpeg has a working h264_nvenc encoder.

    Probed once per process (lru_cache) by attempting a tiny test encode.
    A dry-run encode is used rather than just checking ``-encoders`` output
    because the encoder may be listed but unavailable at runtime (no GPU,
    driver mismatch, etc.).
    """
    try:
        r = subprocess.run(
            [  # noqa: S607
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:d=0.04",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            check=False,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


@functools.lru_cache(maxsize=1)
def _timing_flags() -> tuple[str, ...]:
    """Frame-timing passthrough flags for the local ffmpeg.

    ``-fps_mode passthrough`` (ffmpeg >= 5.0) preserves each frame's timing with no
    drop/dup. ffmpeg 4.x (e.g. the 4.4.2 shipped in some runtime containers) has no
    ``-fps_mode`` and errors out ("Unrecognized option 'fps_mode'"); there the
    equivalent is the older ``-vsync 0``, which also works on newer ffmpeg. Probe once
    (cached) so the cutter is correct on both. Result is process-wide via lru_cache.
    """
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",  # noqa: S607
             "-i", "color=c=black:s=16x16:d=0.04", "-fps_mode", "passthrough", "-f", "null", "-"],
            capture_output=True, text=True, check=False, timeout=30, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        if probe.returncode == 0:
            return ("-fps_mode", "passthrough")
    except (OSError, subprocess.SubprocessError):
        pass
    return ("-vsync", "0")


def _ffprobe(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ffprobe with automatic retry on transient failures, returning the completed process."""
    # Retry a failed probe a couple of times: probes read the source (possibly over a
    # network mount / S3, or sharing a node's disk/CPU with other concurrent probes), so
    # a transient blip -- a nonzero return *or* a timeout -- should not fail the whole
    # chunk outright. Only a timeout that persists across every retry propagates.
    cmd = ["ffprobe", "-v", "error", *args]
    attempt = 1
    while True:
        try:
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=False, timeout=_PROBE_TIMEOUT_S, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired:
            if attempt > _PROBE_RETRIES:
                raise
            time.sleep(0.5 * attempt)
            attempt += 1
            continue
        if result.returncode == 0 or attempt > _PROBE_RETRIES:
            return result
        time.sleep(0.5 * attempt)
        attempt += 1


class _Index:
    """Per-frame PTS index (presentation order), keyframe positions, and cut-mode safety."""

    def __init__(  # noqa: PLR0913
        self,
        num: int,
        den: int,
        pts: list[int],
        *,
        stream_copy_safe: bool,
        keyframes: tuple[int, ...] = (),
        smart_cut_safe: bool = False,
    ) -> None:
        self.num = num
        self.den = den
        self.pts = pts  # frame i -> PTS ticks, presentation order
        self.stream_copy_safe = stream_copy_safe
        # Keyframe frame indices in presentation order, mapped by PTS so they stay valid under
        # B-frame reordering. Decode seeks may use these; packet COPYING still needs
        # smart_cut_safe, since that depends on file order rather than presentation order.
        self.keyframes = keyframes
        self.smart_cut_safe = smart_cut_safe

    def __len__(self) -> int:
        return len(self.pts)

    def next_keyframe_at_or_after(self, frame: int) -> int | None:
        """Return first keyframe index >= *frame*, or None if there is none at/after it."""
        i = bisect.bisect_left(self.keyframes, frame)
        return self.keyframes[i] if i < len(self.keyframes) else None

    def keyframe_at_or_before(self, frame: int) -> int | None:
        """Return the last keyframe index <= *frame*, or None when there is none.

        A re-encode decodes from here so the frames *frame* depends on are present. Seeking at
        the frame's own timestamp instead loses the reference on MPEG-TS, where the demuxer
        starts at that packet and decoding silently resumes at the NEXT keyframe.
        """
        i = bisect.bisect_right(self.keyframes, frame) - 1
        return self.keyframes[i] if i >= 0 else None


def _is_one_packet_per_frame(codec_name: str, field_order: str, format_name: str) -> bool:
    """Return True when the container/codec guarantees one displayed frame per packet."""
    formats = {f.strip() for f in format_name.split(",") if f.strip()}
    return (
        bool(formats & _ONE_PACKET_PER_FRAME_FORMATS)
        and codec_name in _ONE_PACKET_PER_FRAME_CODECS
        and field_order not in _INTERLACED_FIELD_ORDERS
    )


def _displayed_frame_count(source: str) -> int:
    """Return the decoded (DISPLAYED) frame count -- the only sound reference for the packet index.

    Must come from ``nb_read_frames`` (a real decode). The container's ``nb_frames`` is the
    moov sample count, i.e. the PACKET count, so checking against it would be tautological --
    and it disagrees with the displayed count on edit-listed sources, where it counts the
    DISCARD pre-roll that this index correctly skips.
    """
    r = _ffprobe(
        ["-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nw=1:nk=1", source]
    )  # fmt: skip
    if r.returncode != 0:
        msg = f"displayed-frame probe failed for {source!r}: {r.stderr.strip()[-300:]}"
        raise ValueError(msg)
    lines = r.stdout.strip().splitlines()
    val = lines[0].strip() if lines else ""
    if not val.isdigit():
        msg = f"{source!r}: ffprobe reported no displayed frame count ({val!r})"
        raise ValueError(msg)
    return int(val)


def _prefix_packet_and_frame_counts(source: str, seconds: int) -> tuple[int, int]:
    """Return non-DISCARD packets and DISPLAYED frames over the first *seconds* of *source*.

    Both probes use the same ``-read_intervals`` window, so the counts are directly
    comparable, and only that window is decoded -- orders of magnitude cheaper than
    decoding the whole file while still exposing a systematic packets-per-frame violation.
    """
    window = f"%+{seconds}"
    pk = _ffprobe(
        ["-read_intervals", window, "-select_streams", "v:0", "-show_packets",
         "-show_entries", "packet=flags", "-of", "csv=p=0", source]
    )  # fmt: skip
    fr = _ffprobe(
        ["-read_intervals", window, "-select_streams", "v:0", "-show_frames",
         "-show_entries", "frame=pts", "-of", "csv=p=0", source]
    )  # fmt: skip
    if pk.returncode != 0 or fr.returncode != 0:
        err = (pk.stderr or fr.stderr or "").strip()[-300:]
        msg = f"prefix index probe failed for {source!r}: {err}"
        raise ValueError(msg)
    # Only the FIRST csv field is the flags string: some containers append side-data columns
    # (MPEG-TS emits "K__,MPEGTS Stream ID,224"), and matching "D" against the whole line would
    # treat the "ID" in that label as a DISCARD flag and drop almost every packet.
    n_pk = 0
    for line in pk.stdout.splitlines():
        flags = line.split(",", 1)[0].strip()
        if flags and "D" not in flags:
            n_pk += 1
    n_fr = sum(1 for line in fr.stdout.splitlines() if line.strip())
    return n_pk, n_fr


def _unambiguous_keyframes(pts: list[int], keyframes: tuple[int, ...]) -> tuple[int, ...]:
    """Keep only keyframes whose timestamp is unique, so seeking to one is unambiguous."""
    return tuple(
        frame
        for frame in keyframes
        if (frame == 0 or pts[frame] != pts[frame - 1]) and (frame + 1 == len(pts) or pts[frame] != pts[frame + 1])
    )


def _probe_index(source: str, verify_index: str = "auto") -> _Index:  # noqa: C901, PLR0912, PLR0915
    """Build the PTS index from container packets (no decode).

    Frame *k* is taken to be the *k*-th non-DISCARD packet in presentation order. That holds
    only when the container stores exactly one displayed frame per packet -- true for
    progressive H.264/HEVC in ISOBMFF (one sample = one access unit = one coded picture), but
    NOT in general: split AV1 temporal units, VP9 superframes, field-coded pictures and some
    remuxes carry several (or zero) displayed frames per packet.

    A wrong index is NOT caught downstream. Every cut mode seeks to ``index.pts[start]`` and
    ffmpeg is told to emit exactly ``expected`` frames, so a mis-mapped index produces a clip
    with the CORRECT frame count and the WRONG content; the per-cut frame-count check only
    detects truncation. The index cardinality is therefore verified here:

      * ``auto`` (default) -- no check for layouts proven 1:1 (progressive H.264/HEVC in ISOBMFF);
        every other layout, including the AV1 sources here, is checked over a bounded prefix
        (~0.5s, vs ~95-115s to decode a whole AV1 source);
      * ``always`` -- decode the WHOLE file and compare (CI and per-dataset audits);
      * ``never`` -- trust the packet index (only for a layout already verified).

    A mismatch raises: no cut mode is safe against a bad index, so ``cut_plan`` propagates it
    and the caller skips the whole chunk rather than emit silently shifted clips.

    Stream-copy is enabled only when the source is all-intra AND packets are already in
    presentation order (no B-frame reordering), so copying packets in file order yields
    exactly the requested frames.
    """
    if verify_index not in _VERIFY_INDEX_MODES:
        msg = f"verify_index must be one of {sorted(_VERIFY_INDEX_MODES)}; got {verify_index!r}"
        raise ValueError(msg)
    meta = _ffprobe(
        ["-select_streams", "v:0", "-show_entries",
         "stream=time_base,codec_name,field_order:format=format_name", "-of", "json", source]
    )  # fmt: skip
    if meta.returncode != 0:
        msg = f"ffprobe failed for {source!r}: {meta.stderr.strip()[-300:]}"
        raise ValueError(msg)
    try:
        meta_json = json.loads(meta.stdout or "{}")
    except json.JSONDecodeError as e:
        msg = f"ffprobe returned unparsable metadata for {source!r}: {e}"
        raise ValueError(msg) from e
    # Containers that expose a "program" (MPEG-TS/HLS) make ffprobe emit the stream twice; the
    # entries are identical, so take the first.
    streams = meta_json.get("streams") or []
    if not streams:
        msg = f"{source!r} has no video stream"
        raise ValueError(msg)
    stream0 = streams[0]
    tb_val = str(stream0.get("time_base") or "").strip()
    if "/" not in tb_val:
        msg = f"{source!r} has no usable time_base ({tb_val!r})"
        raise ValueError(msg)
    num, den = (int(x) for x in tb_val.split("/", 1))
    if num <= 0 or den <= 0:
        msg = f"{source!r} has non-positive time_base {num}/{den}"
        raise ValueError(msg)
    codec_name = str(stream0.get("codec_name") or "")
    field_order = str(stream0.get("field_order") or "")
    format_name = str((meta_json.get("format") or {}).get("format_name") or "")

    pk = _ffprobe(
        ["-select_streams", "v:0", "-show_packets", "-show_entries", "packet=pts,flags", "-of", "json", source]
    )
    if pk.returncode != 0:
        msg = f"packet index probe failed for {source!r}: {pk.stderr.strip()[-300:]}"
        raise ValueError(msg)
    packets = json.loads(pk.stdout).get("packets", [])
    file_pts: list[int] = []
    keyframe_rows: list[int] = []
    all_intra = True
    for row, p in enumerate(packets):
        flags = p.get("flags") or ""
        if "D" in flags:
            # AV_PKT_FLAG_DISCARD: edit-list pre-roll (e.g. copy-trimmed sources) that is decoded
            # but NOT displayed. Counting it would inflate the index and shift every cut, so skip it
            # — the index must match display order, which is what plan frame numbers reference.
            continue
        val = p.get("pts")
        if val in (None, "N/A"):
            msg = f"{source!r} packet {row} has no pts"
            raise ValueError(msg)
        if "K" in flags:
            keyframe_rows.append(len(file_pts))  # this frame's index in file/decode order
        else:
            all_intra = False
        file_pts.append(int(val))
    if not file_pts:
        msg = f"{source!r} produced an empty packet index"
        raise ValueError(msg)

    file_monotonic = all(file_pts[i] > file_pts[i - 1] for i in range(1, len(file_pts)))
    pts = sorted(file_pts)
    duplicates = sum(1 for i in range(1, len(pts)) if pts[i] == pts[i - 1])
    if duplicates:
        # VFR sources can legitimately repeat a PTS; do NOT drop the whole chunk. Instead
        # stream-copy is disabled and any cut that *starts* on a duplicated (ambiguous)
        # frame is failed per-cut in _cut_one -- a shifted clip would keep the right frame
        # count and slip past validation, so we must not attempt it.
        logger.warning(f"{source!r}: {duplicates} duplicate packet PTS; cuts starting on an ambiguous frame will fail")
    # Smart-cut needs packets already in presentation order and unambiguous PTS (so keyframe file
    # indices are valid frame indices and a per-clip copy lands on the right frame); it does NOT
    # need all-intra (that is the stronger whole-file stream-copy condition).
    smart_cut_safe = file_monotonic and duplicates == 0

    # Cardinality check: is the k-th packet really the k-th DISPLAYED frame? Nothing downstream
    # can catch a violation (a shifted clip keeps the requested frame count), so verify it here.
    # Skipped for layouts that guarantee 1:1 -- which covers the H.264/mp4 production sources at
    # zero cost, since the decode this needs is far more expensive than the packet probe.
    def _reject(n_pkts: int, n_frames: int, scope: str) -> None:
        msg = (
            f"{source!r}: {n_pkts} packet(s) but {n_frames} displayed frame(s) {scope}, so packet "
            f"positions are NOT frame indices for this codec/container "
            f"({codec_name or '?'} in {format_name or '?'}); every cut would be content-shifted "
            f"while still producing the requested frame count"
        )
        raise ValueError(msg)

    if verify_index == "always":
        displayed = _displayed_frame_count(source)
        if displayed != len(pts):
            _reject(len(pts), displayed, "over the whole file")
    elif verify_index == "auto" and not _is_one_packet_per_frame(codec_name, field_order, format_name):
        n_pk, n_fr = _prefix_packet_and_frame_counts(source, _PREFIX_VERIFY_SECONDS)
        if n_pk != n_fr:
            _reject(n_pk, n_fr, f"in the first {_PREFIX_VERIFY_SECONDS}s")
    return _Index(
        num,
        den,
        pts,
        stream_copy_safe=all_intra and smart_cut_safe,
        # Mapped through PTS so the indices are presentation-order positions even when packets
        # are in decode order; duplicate timestamps invalidate only their own seek positions.
        keyframes=_unambiguous_keyframes(
            pts, tuple(sorted(bisect.bisect_left(pts, file_pts[row]) for row in keyframe_rows))
        ),
        smart_cut_safe=smart_cut_safe,
    )


def _seconds(ticks: float, num: int, den: int) -> str:
    """Convert PTS ticks to a seconds string (9 dp; sub-microsecond, finer than any frame)."""
    return f"{ticks * num / den:.9f}"


def _keyframe_seek_seconds(pts: int, num: int, den: int) -> str:
    """Return an ``-ss`` value that lands at or before the keyframe at *pts*.

    ffmpeg parses ``-ss`` into whole microseconds, discarding finer digits, then rescales
    that to source ticks with nearest rounding. Flooring to a whole microsecond here means
    the rounding can only pull the seek earlier, never past the keyframe -- landing past it
    would start decoding without the reference the requested frames depend on, which is how
    MPEG-TS cuts silently resumed at the FOLLOWING keyframe. Seeking early is free of
    correctness risk and costs only decode work: ``trim=start_pts`` still selects the exact
    frames, so the seek never decides which frames are kept.
    """
    micros = pts * num * 1_000_000 // den
    return f"{micros // 1_000_000}.{micros % 1_000_000:06d}"


def _encode_cmd(  # noqa: PLR0913
    source: str,
    output: str,
    index: _Index,
    start: int,
    nframes: int,
    bitrate: str,
    threads: int,
    *,
    gpu: bool = False,
) -> list[str]:
    """Build the ffmpeg re-encode command for a single clip.

    Decodes from a keyframe at or before *start* and selects the range by integer source PTS.
    The input seek only bounds decode work -- it is deliberately inexact -- while the ``trim``
    owns which frames are kept, so a seek that lands earlier than predicted cannot shift the
    clip. Seeking at the frame's own timestamp instead loses the reference keyframe on MPEG-TS.

    When *gpu* is True the NVIDIA NVENC hardware encoder (``h264_nvenc``) is
    used instead of libopenh264, which offloads encoding to the GPU's dedicated
    video-encode engine and is typically 5-10x faster.  Falls back to
    libopenh264 if h264_nvenc is not available (callers should guard with a
    capability check before setting gpu=True).
    """
    use_nvenc = gpu and _nvenc_available()
    if gpu and not use_nvenc:
        logger.warning("h264_nvenc requested but not available; falling back to libopenh264")
    encoder = "h264_nvenc" if use_nvenc else "libopenh264"
    # h264_nvenc uses -rc cbr for constant bitrate; libopenh264 uses -b:v directly.
    extra: list[str] = ["-rc", "cbr"] if use_nvenc else []
    # Seek to a keyframe at/before start, floored to whole microseconds so rounding can only
    # land earlier. With no trusted keyframe, decode from the beginning rather than guess.
    seek: list[str] = []
    kf = index.keyframe_at_or_before(start)
    if kf is not None and index.pts[kf] >= 0:
        seek = ["-ss", _keyframe_seek_seconds(index.pts[kf], index.num, index.den)]
    # settb puts the filter graph on the source timebase so start_pts is in source ticks.
    # showinfo logs each selected frame's source PTS BEFORE setpts discards it, which is what
    # _selected_source_pts_match checks -- a same-length substitution passes a frame count.
    vf = (
        f"settb=expr={index.num}/{index.den},trim=start_pts={index.pts[start]},"
        f"trim=end_frame={nframes},showinfo@source=checksum=0,setpts=PTS-STARTPTS"
    )
    return [
        "ffmpeg", "-y", "-nostdin", "-nostats", "-v", "info", "-copyts", "-seek_timestamp", "1",
        *seek, "-noaccurate_seek", "-i", source, "-map", "0:v:0", "-an",
        "-frames:v", str(nframes), "-vf", vf,
        "-c:v", encoder, *extra, "-b:v", bitrate, "-pix_fmt", "yuv420p",
        *([] if use_nvenc else ["-threads", str(threads)]),
        *_timing_flags(), output,
    ]  # fmt: skip


def _copy_cmd(source: str, output: str, ss: str, nframes: int) -> list[str]:
    """Build the ffmpeg stream-copy command for a single clip."""
    # Stream copy: seek onto the (keyframe) start frame and copy exactly nframes packets.
    # Safe only for all-intra sources with in-order packets (see _probe_index).
    return [
        "ffmpeg", "-y", "-nostdin", "-v", "error", "-seek_timestamp", "1",
        "-ss", ss, "-i", source, "-map", "0:v:0", "-an",
        "-frames:v", str(nframes), "-c", "copy", "-avoid_negative_ts", "make_zero", output,
    ]  # fmt: skip


def _output_frame_count(path: str) -> int:
    """Return the frame count of the output file, preferring fast container metadata over a full decode."""
    # Prefer the container's nb_frames (no decode); it is written accurately for the mp4
    # we just produced. Fall back to an exact decode count only if it is unavailable.
    fast = _ffprobe(
        ["-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "default=nk=1:nw=1", path]
    ).stdout.strip()
    if fast.isdigit() and int(fast) > 0:
        return int(fast)
    out = _ffprobe(
        ["-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nk=1:nw=1", path]
    ).stdout.strip()  # fmt: skip
    return int(out) if out.isdigit() else -1


def _run_ff(cmd: list[str], errf: IO[bytes]) -> bool:
    """Run an ffmpeg/ffprobe command, capturing stderr to *errf*. True iff exit code 0."""
    return (
        subprocess.run(  # noqa: S603
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errf, timeout=_CUT_TIMEOUT_S, check=False
        ).returncode
        == 0
    )


def _error_tail(errf: IO[bytes], limit: int = 600) -> str:
    """Return the tail of the ffmpeg log without the per-frame ``showinfo`` trace.

    The encode runs at ``-v info`` so the source PTS are observable (see :func:`_encode_cmd`).
    That trace is one line per selected frame, so an unfiltered tail would be nothing but
    frame dumps and would cut off the ffmpeg error that explains the failure.
    """
    errf.seek(0)
    text = errf.read().decode(errors="replace")
    return "\n".join(line for line in text.splitlines() if "showinfo@source" not in line).strip()[-limit:]


def _selected_source_pts_match(errf: IO[bytes], log_start: int, expected_pts: list[int]) -> bool:
    """Verify the decoded frames' source PTS before lossy encoding and the timestamp reset.

    A wrong-but-same-length selection passes every frame count and timestamp check, so identity
    has to be confirmed while the source PTS are still observable -- which is what the showinfo
    pass in :func:`_encode_cmd` logs.
    """
    errf.seek(log_start)
    selected: list[int] = []
    for line in errf:
        if b"showinfo@source" in line:
            m = re.search(rb"\bn:\s*\d+\s+pts:\s*(-?\d+)", line)
            if m is not None:
                selected.append(int(m.group(1)))
    errf.seek(0, 2)
    return selected == expected_pts


def _run_encode(  # noqa: PLR0913
    source: str,
    output: str,
    index: _Index,
    start: int,
    nframes: int,
    bitrate: str,
    threads: int,
    errf: IO[bytes],
    *,
    gpu: bool = False,
) -> bool:
    """Re-encode [start, start+nframes-1], rejecting the run if the wrong frames were selected."""
    log_start = errf.tell()
    if not _run_ff(_encode_cmd(source, output, index, start, nframes, bitrate, threads, gpu=gpu), errf):
        return False
    if not _selected_source_pts_match(errf, log_start, index.pts[start : start + nframes]):
        errf.write(b"\nDecoded source PTS do not match the requested frame range\n")
        return False
    return True


def _smartcut(  # noqa: PLR0913
    source: str,
    output: str,
    start: int,
    kf: int,
    end: int,
    index: _Index,
    bitrate: str,
    threads: int,
    errf: IO[bytes],
    *,
    gpu: bool = False,
    tmp_dir: str | None = None,
) -> bool:
    """Frame-exact smart cut of frames [start, end] when *kf* (a keyframe) satisfies start < kf <= end.

    Re-encode ONLY the head [start, kf-1] -- those frames sit before the first keyframe inside the
    clip, so they depend on frames before ``start`` and cannot be copied -- and STREAM-COPY the body
    [kf, end] bit-exact (kf is a keyframe, so the copied range is self-contained). Both segments
    begin on an IDR, so they concatenate cleanly. Returns True on a clean run; the CALLER still
    validates the output frame count and falls back to a full re-encode on any mismatch, so a bad
    concat can never ship a wrong clip.
    """
    with tempfile.TemporaryDirectory(dir=tmp_dir) as _tmpdir:
        head = str(Path(_tmpdir) / "head.mp4")
        body = str(Path(_tmpdir) / "body.mp4")
        lst = str(Path(_tmpdir) / "concat.txt")
        # Head: re-encode [start, kf-1], decoding from an earlier keyframe and selecting by PTS.
        if not _run_encode(source, head, index, start, kf - start, bitrate, threads, errf, gpu=gpu):
            return False
        # Body: stream-copy [kf, end]. kf is a keyframe, so +0.25 tick lands exactly on it.
        body_ss = _seconds(index.pts[kf] + 0.25, index.num, index.den)
        if not _run_ff(_copy_cmd(source, body, body_ss, end - kf + 1), errf):
            return False
        with Path(lst).open("w") as fh:
            # Escape single quotes per the concat-demuxer syntax ('\'' inside a quoted path);
            # paths are program-generated so this is belt-and-suspenders, and a broken manifest
            # would only fail-safe into the full re-encode fallback anyway.
            def _q(p: str) -> str:
                return str(Path(p).resolve()).replace("'", "'\\''")

            fh.write(f"file '{_q(head)}'\nfile '{_q(body)}'\n")
        concat_cmd = [
            "ffmpeg", "-y", "-nostdin", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", lst, "-c", "copy", "-avoid_negative_ts", "make_zero", output,
        ]  # fmt: skip
        return _run_ff(concat_cmd, errf)


def _cut_one(  # noqa: C901, PLR0912, PLR0913
    source: str,
    cut: dict[str, Any],
    index: _Index,
    bitrate: str,
    threads: int,
    *,
    allow_stream_copy: bool,
    smart_cut: bool = False,
    gpu: bool = False,
    tmp_dir: str | None = None,
) -> dict[str, Any]:
    start, end, output = cut["startFrame"], cut["endFrame"], cut["output"]
    expected = end - start + 1
    rec: dict[str, Any] = {
        "output": output,
        "expected_frames": expected,
        "written_frames": 0,
        "success": False,
        "error": None,
        "mode": None,
    }
    if end >= len(index):
        rec["error"] = f"endFrame {end} beyond source length {len(index)}"
        return rec

    start_pts = index.pts[start]
    if start_pts < 0:
        # Input -ss cannot seek to a negative timestamp (ffmpeg clamps it to 0), which
        # would silently cut the wrong frames. Fail this cut rather than mis-cut it.
        rec["error"] = f"start frame {start} has negative PTS {start_pts}; cannot seek accurately"
        return rec
    pts = index.pts
    if (start > 0 and pts[start] == pts[start - 1]) or (start + 1 < len(pts) and pts[start] == pts[start + 1]):
        # A duplicated (non-unique) start PTS makes the -ss seek ambiguous: it would
        # silently shift the clip while keeping the right frame count. Fail it instead.
        rec["error"] = f"start frame {start} has an ambiguous (duplicated) PTS {start_pts}; cannot seek accurately"
        return rec

    use_copy = allow_stream_copy and index.stream_copy_safe
    # Smart-cut candidate keyframe: the first keyframe at/after start (only when enabled + safe).
    # Smart cut's body is a STREAM COPY, so honor allow_stream_copy=False (--no-stream-copy) by
    # disabling it and falling through to a full re-encode.
    kf = (
        index.next_keyframe_at_or_after(start)
        if (smart_cut and allow_stream_copy and index.smart_cut_safe and index.keyframes)
        else None
    )
    try:
        with tempfile.TemporaryFile() as errf:
            Path(output).parent.mkdir(parents=True, exist_ok=True)

            # Fast paths (validated by frame count below; ANY miss falls back to a full re-encode, so a
            # bad copy/concat can never ship a wrong clip):
            #   - whole-clip stream-copy: an all-intra file, OR the clip starts exactly on a keyframe;
            #   - smart-cut: re-encode only the pre-keyframe head, stream-copy the rest.
            if use_copy or kf == start:
                # Seek a QUARTER tick past start so it re-quantizes back to exactly `start`'s keyframe
                # PTS (+0.5 would round up onto the next keyframe and shift the clip; +0.25 rounds down
                # to `start` at any frame spacing). `start` is a keyframe here, so the copy is exact.
                rec["mode"] = "copy" if use_copy else "smartcopy"
                ok = _run_ff(
                    _copy_cmd(source, output, _seconds(start_pts + 0.25, index.num, index.den), expected), errf
                )
            elif kf is not None and kf <= end:
                rec["mode"] = "smartcut"
                ok = _smartcut(source, output, start, kf, end, index, bitrate, threads, errf, gpu=gpu, tmp_dir=tmp_dir)
            else:
                ok = False  # no fast path applies (or none enabled) -> full re-encode

            if ok:
                rec["written_frames"] = _output_frame_count(output)
                if rec["written_frames"] == expected:
                    rec["success"] = True
                    return rec

            # Full re-encode: the default path AND the frame-exact fallback for any fast-path miss.
            rec["mode"] = "encode"
            # Decode from an earlier keyframe and select [start, end] by integer source PTS.
            if not _run_encode(source, output, index, start, expected, bitrate, threads, errf, gpu=gpu):
                rec["error"] = f"ffmpeg re-encode failed: {_error_tail(errf)}"
                return rec
            rec["written_frames"] = _output_frame_count(output)
            if rec["written_frames"] != expected:
                rec["error"] = f"wrote {rec['written_frames']}/{expected} frames"
            else:
                rec["success"] = True
    except subprocess.TimeoutExpired:
        rec["error"] = "ffmpeg/ffprobe timed out"
    except OSError as e:
        # A per-cut failure (e.g. unwritable output dir / temp file) is isolated to this cut.
        rec["error"] = f"failed to run cut: {e}"
    return rec


def cut_plan(  # noqa: C901, PLR0913
    source: str,
    cuts: list[Any],
    bitrate: str = _DEFAULT_BITRATE,
    workers: int = 1,
    threads: int = 1,
    *,
    allow_stream_copy: bool = True,
    smart_cut: bool = False,
    verify_index: str = "auto",
    gpu: bool = False,
    tmp_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Cut every range in *cuts* from *source*, one independent ffmpeg per cut.

    Each cut fast-seeks to its exact start PTS and stream-copies (all-intra source) or
    re-encodes with libopenh264 (CPU) or h264_nvenc (GPU, when *gpu* is True) at the
    given *bitrate*. Returns one record per cut:
    ``{"output", "expected_frames", "written_frames", "success", "error", "mode"}``;
    ``success`` is True only when the exact requested frame count was produced. A per-cut
    failure is isolated to that cut.

    ``workers`` runs that many cuts concurrently; ``threads`` caps each ffmpeg's threads
    (default 1 each -> minimal per-worker CPU/RAM, for packing many pipeline workers).

    ``verify_index`` ("auto"/"always"/"never") controls the one-packet-per-displayed-frame
    check on *source* (see :func:`_probe_index`); a violation raises rather than silently
    shifting every cut from this source.

    Raises ``ValueError`` for an invalid/duplicate cut, an unreadable source, or a source
    whose packet index is not a frame index.
    """
    if not cuts:
        return []
    seen: set[str] = set()
    for c in cuts:
        if not isinstance(c, dict):
            msg = f"each cut must be an object, got {type(c).__name__}: {c!r}"
            raise TypeError(msg)
        missing = [k for k in ("startFrame", "endFrame", "output") if k not in c]
        if missing:
            msg = f"cut missing required key(s) {missing}: {c}"
            raise ValueError(msg)
        if not isinstance(c["startFrame"], int) or not isinstance(c["endFrame"], int):
            msg = f"startFrame/endFrame must be integers: {c}"
            raise TypeError(msg)
        if not isinstance(c["output"], str):
            msg = f"output must be a string: {c}"
            raise TypeError(msg)
        if c["startFrame"] < 0 or c["endFrame"] < c["startFrame"]:
            msg = f"invalid cut range: {c}"
            raise ValueError(msg)
        if c["output"] in seen:
            msg = f"duplicate output path in plan: {c['output']!r}"
            raise ValueError(msg)
        seen.add(c["output"])

    index = _probe_index(source, verify_index=verify_index)

    def run(c: dict[str, Any]) -> dict[str, Any]:
        return _cut_one(
            source,
            c,
            index,
            bitrate,
            threads,
            allow_stream_copy=allow_stream_copy,
            smart_cut=smart_cut,
            gpu=gpu,
            tmp_dir=tmp_dir,
        )

    if workers <= 1:
        return [run(c) for c in cuts]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, cuts))


def _load_plan(plan_path: Path) -> dict[str, Any]:
    with plan_path.open() as f:
        plan = json.load(f)
    if not isinstance(plan, dict) or "source" not in plan or "cuts" not in plan:
        msg = f"plan must be a JSON object with 'source' and 'cuts': {plan_path}"
        raise ValueError(msg)
    return plan


def main(argv: list[str] | None = None) -> int:
    """Run the frame-exact per-cut video cutter from a JSON cut plan."""
    parser = argparse.ArgumentParser(description="Frame-exact per-cut video cutter (PTS-seek + stream-copy).")
    parser.add_argument("plan", type=Path, help="JSON cut plan (source + inclusive frame ranges).")
    parser.add_argument(
        "--bitrate",
        type=str,
        default=_DEFAULT_BITRATE,
        help="Target bitrate for libopenh264 re-encoded cuts (default: 2M).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent cuts (default: 1).")
    parser.add_argument("--threads", type=int, default=1, help="ffmpeg threads per cut (default: 1).")
    parser.add_argument("--no-stream-copy", action="store_true", help="Always re-encode (never stream-copy).")
    parser.add_argument(
        "--verify-index",
        choices=sorted(_VERIFY_INDEX_MODES),
        default="auto",
        help=(
            "Verify that each packet is one displayed frame before cutting. 'auto' (default) "
            "decodes only for codec/containers not proven 1:1; 'always' decodes every source; "
            "'never' trusts the packet index. A violation fails the source instead of silently "
            "shifting every cut."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Retained for compatibility; outputs are always validated (exact frame count).",
    )
    args = parser.parse_args(argv)

    try:
        plan = _load_plan(args.plan)
    except (OSError, ValueError):
        logger.exception(f"cannot load plan {args.plan}")
        return 1
    cuts = plan["cuts"]
    if not cuts:
        logger.info(f"plan has no cuts: {args.plan}")
        return 0

    logger.info(f"cutting {len(cuts)} range(s) from {plan['source']}")
    try:
        records = cut_plan(
            plan["source"],
            cuts,
            bitrate=args.bitrate,
            workers=args.workers,
            threads=args.threads,
            allow_stream_copy=not args.no_stream_copy,
            verify_index=args.verify_index,
        )
    except (ValueError, OSError, subprocess.SubprocessError):
        logger.exception("cannot run plan")
        return 1

    failures = 0
    for r in records:
        name = Path(r["output"]).name
        if r["success"]:
            logger.info(f"  {name} frames={r['written_frames']}/{r['expected_frames']} ok ({r['mode']})")
        else:
            failures += 1
            logger.error(f"  {name} frames={r['written_frames']}/{r['expected_frames']} FAILED: {r['error']}")
    if failures:
        logger.error(f"{failures} of {len(records)} cut(s) failed")
        return 1
    logger.info(f"complete: {len(records)} cut(s) ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
