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
# ruff: noqa: T201
"""Decode benchmarks for the sensor library.

Subcommands
-----------
sol     Speed-of-light: raw PyAV streaming decode, frames discarded immediately.
        Establishes the fastest possible decode rate for a given video and
        threading configuration.

kf-chunked   Keyframe-chunked: seek+flush per GOP, decode every frame and discard.
        Same total work as sol; exposes the overhead of N seeks and buffer
        flushes (one per keyframe/GOP) vs. a single linear stream.

sensor  CameraSensor-based: divides the video into non-overlapping segments and
        decodes each via CameraSensor.sample.  Exercises the full production decode
        path including VideoIndex construction, grid sampling, and decode plan
        execution.  Validates that every VideoIndex timestamp is decoded exactly once.

Run:
    python -m cosmos_curator.core.sensors.scripts.camera_sensor_benchmark sol --source video.mp4
    python -m cosmos_curator.core.sensors.scripts.camera_sensor_benchmark kf-chunked --source video.mp4
    python -m cosmos_curator.core.sensors.scripts.camera_sensor_benchmark sensor --source video.mp4
    python -m cosmos_curator.core.sensors.scripts.camera_sensor_benchmark sensor \
        --source video.mp4 --segment-duration 30
"""

import argparse
import sys
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, cast

import av
import av.codec.context
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from cosmos_curator.core.sensors.data.camera_data import CameraData
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.scripts._cli_cloud import (
    CloudCliError,
    add_cloud_credential_args,
    is_cloud_uri,
    open_cloud_source,
    validate_source,
)
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.types.types import DataSource, VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.io import open_data_source
from cosmos_curator.core.sensors.utils.video import (
    CpuVideoDecodeConfig,
    make_decode_plan,
    make_index_and_metadata,
    open_video_container,
)

# A factory that produces a fresh ``DataSource`` context per call. Used so each
# library entry-point (CameraSensor, make_index_and_metadata, _verify_segment) gets
# its own stream rather than aliasing one shared BinaryIO, which the new sensor
# library contract forbids (concurrent / overlapping use is unsupported).
SourceFactory = Callable[[], AbstractContextManager[DataSource]]


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    """Attach common source / VideoIndex arguments to *parser*."""
    parser.add_argument(
        "--source",
        required=True,
        help="Video source: local file path, or s3:// / az:// URI.",
    )
    parser.add_argument(
        "--full-demux-index",
        action="store_true",
        help="Build VideoIndex via full demux instead of container header (slower, for validation).",
    )
    add_cloud_credential_args(parser)


def _resolve_source(args: argparse.Namespace) -> tuple[str, SourceFactory]:
    """Validate ``args.source`` and return a (label, source-factory) pair.

    The factory is a thunk returning a context manager that yields a fresh
    :class:`DataSource` each time: a :class:`Path` for local files, a fresh
    seekable :class:`BinaryIO` for cloud URIs. Exits with status 1 on invalid
    source / credential errors.
    """
    source_str: str = args.source
    try:
        validate_source(source_str)
    except CloudCliError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)

    if is_cloud_uri(source_str):
        s3_profile_name = args.s3_profile_name
        azure_profile_name = args.azure_profile_name

        @contextmanager
        def factory() -> Generator[DataSource, None, None]:
            with open_cloud_source(
                source_str,
                s3_profile_name=s3_profile_name,
                azure_profile_name=azure_profile_name,
            ) as stream:
                yield cast("DataSource", stream)

        return source_str, factory

    local_path = Path(source_str)

    @contextmanager
    def local_factory() -> Generator[DataSource, None, None]:
        yield local_path

    return source_str, local_factory


def _add_threading_args(
    parser: argparse.ArgumentParser,
    *,
    thread_type_default: str | None = "NONE",
    thread_count_default: int | None = 0,
) -> None:
    """Attach codec threading arguments to *parser*."""
    thread_type_default_str = thread_type_default if thread_type_default is not None else "decoder default"
    parser.add_argument(
        "--thread-type",
        choices=["NONE", "FRAME", "SLICE", "AUTO"],
        default=thread_type_default,
        help=(
            f"Codec threading model (default: {thread_type_default_str}). "
            "NONE: single-threaded, no parallelism. "
            "FRAME: decode multiple frames in parallel across threads — best throughput for "
            "sequential playback but adds latency (one thread per complete frame). "
            "SLICE: decode independent slices of a single frame in parallel — lower latency "
            "than FRAME, but only effective for codecs that encode with slices (e.g. some H.264 streams). "
            "AUTO: let FFmpeg choose FRAME|SLICE based on the codec."
        ),
    )
    parser.add_argument(
        "--thread-count",
        type=int,
        default=thread_count_default,
        metavar="N",
        help=(
            "Number of decoder threads "
            f"(default: {thread_count_default if thread_count_default is not None else 'decoder default'}"
            f"{' = let FFmpeg choose based on CPU count' if thread_count_default == 0 else ''})."
        ),
    )


def cmd_sol(args: argparse.Namespace) -> None:
    """Speed-of-light: decode every frame with PyAV and discard immediately."""
    source_label, open_source = _resolve_source(args)

    index_method = (
        VideoIndexCreationMethod.FULL_DEMUX if args.full_demux_index else VideoIndexCreationMethod.FROM_HEADER
    )
    print(f"source : {source_label}")
    print("building VideoIndex ...")
    with open_source() as data:
        index, metadata = make_index_and_metadata(data, index_method=index_method)
    duration_s = (index.pts_ns[-1] - index.pts_ns[0]) / 1e9
    print(
        f"  {len(index)} frames  |  {duration_s:.2f} s  |  {float(metadata.avg_frame_rate):.3f} fps"
        f"  |  {metadata.width}x{metadata.height}  |  {metadata.bit_rate_bps // 1000} kbps"
        f"  |  codec={metadata.codec_name} {metadata.codec_profile}  |  max_bframes={metadata.codec_max_bframes}"
        f"  |  pix_fmt={metadata.pix_fmt}"
    )

    thread_type = av.codec.context.ThreadType[args.thread_type]
    print(f"\nthread_type={thread_type.name}  thread_count={args.thread_count or 'auto'}")

    decoded = 0

    with (
        open_source() as data,
        open_data_source(data, mode="rb") as raw_stream,
        open_video_container(cast("Any", raw_stream)) as (container, stream),
    ):
        stream.thread_type = thread_type
        stream.thread_count = args.thread_count

        print("\ndemuxing + decoding (streaming, split timing) ...")
        t_demux = 0.0
        t_decode = 0.0
        t_convert = 0.0
        pbar = tqdm(total=len(index), unit="frame", desc="demux+decode")
        demux_iter = container.demux(stream)
        t_wall = time.perf_counter()

        while True:
            t0 = time.perf_counter()
            try:
                packet = next(demux_iter)
            except StopIteration:
                break
            t_demux += time.perf_counter() - t0

            t0 = time.perf_counter()
            for frame in packet.decode():
                decoded += 1
                pbar.update(1)
                t_conv = time.perf_counter()
                frame.to_ndarray(format="rgb24")
                t_convert += time.perf_counter() - t_conv
            t_decode += time.perf_counter() - t0

        pbar.close()
        elapsed = time.perf_counter() - t_wall

        fps = decoded / elapsed if elapsed > 0 else float("inf")
        speed = duration_s / elapsed if elapsed > 0 else float("inf")
        print(f"\ndecoded {decoded} frames in {elapsed:.2f} s")
        print(f"  demux      : {t_demux:.2f} s  ({100 * t_demux / elapsed:.1f}%)")
        print(f"  decode     : {t_decode - t_convert:.2f} s  ({100 * (t_decode - t_convert) / elapsed:.1f}%)")
        print(f"  to_ndarray : {t_convert:.2f} s  ({100 * t_convert / elapsed:.1f}%)")
        print(f"  {fps:.1f} fps  |  {speed:.2f}x realtime")


def cmd_kf_chunked(args: argparse.Namespace) -> None:  # noqa: PLR0915
    """Keyframe-chunked decode: seek+flush per GOP, decode every frame, discard immediately.

    Builds a decode plan targeting every frame, then for each GOP:
      1. Seek to the governing keyframe.
      2. Flush codec buffers.
      3. Decode forward through the GOP, discarding all frames.

    Because all frames are targeted there is no sparsity gain — throughput
    should be similar to sol.  The overhead visible here is the cost of N
    seeks and buffer flushes (one per keyframe/GOP).
    """
    source_label, open_source = _resolve_source(args)

    index_method = (
        VideoIndexCreationMethod.FULL_DEMUX if args.full_demux_index else VideoIndexCreationMethod.FROM_HEADER
    )
    print(f"source : {source_label}")
    print("building VideoIndex ...")
    with open_source() as data:
        index, metadata = make_index_and_metadata(data, index_method=index_method)
    duration_s = (index.pts_ns[-1] - index.pts_ns[0]) / 1e9
    n_gops = len(index.kf_pts_ns)
    print(
        f"  {len(index)} frames  |  {duration_s:.2f} s  |  {float(metadata.avg_frame_rate):.3f} fps"
        f"  |  {metadata.width}x{metadata.height}  |  {metadata.bit_rate_bps // 1000} kbps"
        f"  |  codec={metadata.codec_name} {metadata.codec_profile}  |  max_bframes={metadata.codec_max_bframes}"
        f"  |  pix_fmt={metadata.pix_fmt}  |  {n_gops} GOPs"
    )

    thread_type = av.codec.context.ThreadType[args.thread_type]
    print(f"\nthread_type={thread_type.name}  thread_count={args.thread_count or 'auto'}")

    counts = np.ones(len(index.pts_stream), dtype=np.int64)
    plan = make_decode_plan(index.kf_pts_stream, index.pts_stream, counts)

    plan_frames = sum(count for _, group in plan for _, count in group)
    if plan_frames != len(index):
        msg = f"Decode plan covers {plan_frames} frames but VideoIndex has {len(index)} — mismatch"
        raise ValueError(msg)

    print(f"\ndecoding {len(plan)} GOPs ...")
    t_seek = 0.0
    t_decode = 0.0
    t_convert = 0.0
    decoded = 0
    seen_pts: set[int] = set()

    with (
        open_source() as data,
        open_data_source(data, mode="rb") as raw_stream,
        open_video_container(cast("Any", raw_stream)) as (container, stream),
        tqdm(total=len(index), unit="frame", desc="kf-chunked") as pbar,
    ):
        stream.thread_type = thread_type
        stream.thread_count = args.thread_count

        t_wall = time.perf_counter()

        for kf_pts_stream, group in plan:
            t0 = time.perf_counter()
            container.seek(kf_pts_stream, stream=stream)
            stream.codec_context.flush_buffers()
            t_seek += time.perf_counter() - t0

            last_target_stream = group[-1][0]

            t0 = time.perf_counter()
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    frame_pts = frame.pts
                    if frame_pts in seen_pts:
                        msg = (
                            f"Duplicate frame pts={frame_pts} decoded twice — "
                            f"GOP kf_pts_stream={kf_pts_stream}, decoded={decoded}"
                        )
                        raise ValueError(msg)
                    seen_pts.add(frame_pts)
                    decoded += 1
                    pbar.update(1)
                    if decoded > len(index):
                        msg = (
                            f"Decoded {decoded} frames but VideoIndex only has {len(index)} — "
                            f"current frame pts={frame_pts}, GOP kf_pts_stream={kf_pts_stream}"
                        )
                        raise ValueError(msg)
                    t_conv = time.perf_counter()
                    _output = frame.to_ndarray(format="rgb24")
                    t_convert += time.perf_counter() - t_conv

                    if frame_pts >= last_target_stream:
                        break
                else:
                    continue
                break
            t_decode += time.perf_counter() - t0

    elapsed = time.perf_counter() - t_wall
    fps = decoded / elapsed if elapsed > 0 else float("inf")
    speed = duration_s / elapsed if elapsed > 0 else float("inf")
    print(f"\ndecoded {decoded} frames in {elapsed:.2f} s")
    print(f"  seek+flush : {t_seek:.2f} s  ({100 * t_seek / elapsed:.1f}%)")
    print(f"  decode     : {t_decode - t_convert:.2f} s  ({100 * (t_decode - t_convert) / elapsed:.1f}%)")
    print(f"  to_ndarray : {t_convert:.2f} s  ({100 * t_convert / elapsed:.1f}%)")
    print(f"  {fps:.1f} fps  |  {speed:.2f}x realtime")


def _verify_segment(
    open_source: SourceFactory,
    camera_data: CameraData,
) -> list[str]:
    """Independently decode each frame via raw PyAV and compare pixel-by-pixel.

    For each frame in *camera_data*, seeks to the exact stream PTS stored in
    ``camera_data.pts_stream``, decodes with a fresh codec context flush, and
    compares the result against ``camera_data.frames[i]``.  Not efficient —
    one seek+flush per frame — but is authoritative.

    Returns a list of per-frame error strings; empty when all frames match.
    """
    errors: list[str] = []
    pts_to_frame_idx = _group_frame_indices_by_pts(camera_data)
    pts_to_find = set(pts_to_frame_idx)

    with (
        open_source() as data,
        open_data_source(data, mode="rb") as raw_stream,
        open_video_container(cast("Any", raw_stream)) as (container, stream),
    ):
        stream.thread_type = "AUTO"
        stream.thread_count = 0
        container.seek(int(camera_data.pts_stream[0]), stream=stream)
        stream.codec_context.flush_buffers()

        for packet in container.demux(stream):
            # Always call decode() — even on the flush packet (size == 0).
            # The flush packet drains B-frames held in the codec output buffer.
            # Without this, the last few B-frames at end-of-stream are never emitted.
            try:
                frames_in_packet = list(packet.decode())
            except av.error.EOFError:
                break
            for frame in frames_in_packet:
                if frame.pts is None:
                    continue
                frame_pts = frame.pts
                if frame_pts in pts_to_find:
                    pts_to_find.remove(frame_pts)
                    decoded_arr = _decode_frame_rgb24(frame)
                    errors.extend(
                        _compare_decoded_frame(camera_data, decoded_arr, frame_pts, pts_to_frame_idx[frame_pts])
                    )
            if packet.size == 0:
                break

    for pts in pts_to_find:
        errors.extend(
            f"frame[{frame_idx}] pts={pts}: stream ended without finding frame" for frame_idx in pts_to_frame_idx[pts]
        )

    return errors


def _group_frame_indices_by_pts(camera_data: CameraData) -> dict[int, list[int]]:
    """Group frame indices by stream PTS, preserving duplicates."""
    pts_to_frame_idx: dict[int, list[int]] = {}
    for i, pts in enumerate(camera_data.pts_stream):
        pts_to_frame_idx.setdefault(int(pts), []).append(i)
    return pts_to_frame_idx


def _decode_frame_rgb24(frame: av.VideoFrame) -> npt.NDArray[np.uint8]:
    """Decode *frame* to a contiguous uint8 RGB array."""
    decoded_arr: npt.NDArray[Any] = frame.to_ndarray(format="rgb24")
    if not decoded_arr.flags.c_contiguous:
        decoded_arr = np.ascontiguousarray(decoded_arr)
    if decoded_arr.dtype != np.uint8:
        decoded_arr = decoded_arr.astype(np.uint8, copy=False)
    return cast("npt.NDArray[np.uint8]", decoded_arr)


def _compare_decoded_frame(
    camera_data: CameraData,
    decoded_arr: npt.NDArray[np.uint8],
    frame_pts: int,
    frame_indices: list[int],
) -> list[str]:
    """Compare one decoded frame against all benchmark rows sharing the same PTS."""
    errors: list[str] = []
    for frame_idx in frame_indices:
        ref = camera_data.frames[frame_idx]
        if not np.array_equal(ref, decoded_arr):
            diff = np.abs(ref.astype(np.int32) - decoded_arr.astype(np.int32))
            errors.append(
                f"frame[{frame_idx}] pts={frame_pts}: max_pixel_diff={diff.max()} mean_diff={diff.mean():.2f}"
            )
    return errors


def cmd_sensor(args: argparse.Namespace) -> None:  # noqa: PLR0915
    """CameraSensor decode benchmark: exercises the full production decode path.

    Divides the video into non-overlapping segments of --segment-duration seconds
    (stride == duration, no overlap) and drives CameraSensor.sample over each one.
    Uses the video's own timestamps as grid points so every frame is targeted
    exactly once.  Raises ValueError if the total decoded frame count does not
    match the VideoIndex timestamp count.
    """
    source_label, open_source = _resolve_source(args)

    segment_ns = int(args.segment_duration * 1_000_000_000)
    default_decode_config = CpuVideoDecodeConfig()
    decode_config = CpuVideoDecodeConfig(
        thread_type=args.thread_type if args.thread_type is not None else default_decode_config.thread_type,
        thread_count=args.thread_count if args.thread_count is not None else default_decode_config.thread_count,
    )

    index_method = (
        VideoIndexCreationMethod.FULL_DEMUX if args.full_demux_index else VideoIndexCreationMethod.FROM_HEADER
    )
    print(f"source           : {source_label}")
    print("building CameraSensor ...")

    # The CameraSensor and its .sample(...) iteration must share one underlying
    # data source; after the refactor we keep one stream open for the whole loop.
    # _verify_segment opens its own independent stream per call to avoid
    # concurrent reuse of the sensor's stream.
    with open_source() as sensor_data:
        sensor = CameraSensor(
            sensor_data,
            index_method=index_method,
            decode_config=decode_config,
        )
        index = sensor.video_index
        video_meta = sensor.video_metadata

        video_duration_s = (index.pts_ns[-1] - index.pts_ns[0]) / 1e9
        n_gops = len(index.kf_pts_ns)
        print(
            f"  {len(index)} frames  |  {video_duration_s:.2f} s  |  {float(video_meta.avg_frame_rate):.3f} fps"
            f"  |  {video_meta.width}x{video_meta.height}  |  {video_meta.bit_rate_bps // 1000} kbps"
            f"  |  codec={video_meta.codec_name} {video_meta.codec_profile}"
            f"  |  max_bframes={video_meta.codec_max_bframes}"
            f"  |  pix_fmt={video_meta.pix_fmt}  |  {n_gops} GOPs"
        )
        print(f"\nsegment-duration : {args.segment_duration:.1f} s  ({segment_ns} ns)")
        print(f"threading        : type={decode_config.thread_type}  count={decode_config.thread_count}")

        # Build sampling grid from the video's own timestamps so each grid point maps to
        # exactly the frame that exists at that time (zero delta, no rounding).
        # A sentinel one nanosecond past the last frame ensures it falls within the
        # half-open interval [grid[0], grid[-1]) that sample_window_indices applies
        # to the final segment — otherwise the last frame of the video would be dropped.
        sentinel = int(index.pts_ns[-1]) + 1

        # stride_ns == duration_ns → non-overlapping segments, no gaps.
        sampling_grid = SamplingGrid(
            start_ns=int(index.pts_ns[0]),
            exclusive_end_ns=sentinel,
            timestamps_ns=index.pts_ns,
            stride_ns=segment_ns,
            duration_ns=segment_ns,
        )
        spec = SamplingSpec(grid=sampling_grid)

        n_segments = sum(1 for _ in sampling_grid)
        print(f"  {n_segments} segments")

        stats: dict[str, float] = {}
        decoded = 0
        verify_errors = 0
        t_verify = 0.0
        t_wall = time.perf_counter()

        with tqdm(total=len(index), unit="frame", desc="sensor") as pbar:
            for camera_data in sensor.sample(spec, stats=stats):
                n = len(camera_data.frames)
                decoded += n
                pbar.update(n)
                if args.verify:
                    t_v0 = time.perf_counter()
                    for err in _verify_segment(open_source, camera_data):
                        print(f"  VERIFY ERROR: {err}", file=sys.stderr)
                        verify_errors += 1
                    t_verify += time.perf_counter() - t_v0

        elapsed = time.perf_counter() - t_wall

    elapsed_decode = elapsed - t_verify  # Exclude verification time from decode metrics

    if decoded != len(index):
        msg = f"CameraSensor decoded {decoded} frames but VideoIndex has {len(index)} timestamps — mismatch"
        raise ValueError(msg)

    t_seek = stats.get("t_seek", 0.0)
    t_convert = stats.get("t_convert", 0.0)
    t_copy = stats.get("t_copy", 0.0)
    t_decode = elapsed_decode - t_seek - t_convert - t_copy
    frames_decoded = int(stats.get("frames_decoded", 0))
    duplicate_frames = frames_decoded - decoded

    fps = decoded / elapsed if elapsed > 0 else float("inf")
    speed = video_duration_s / elapsed if elapsed > 0 else float("inf")
    print(f"\ndecoded {decoded} frames in {elapsed:.2f} s")
    print(f"  seek+flush     : {t_seek:.2f} s  ({100 * t_seek / elapsed:.1f}%)")
    print(f"  decode (libav) : {t_decode:.2f} s  ({100 * t_decode / elapsed:.1f}%)")
    print(f"  to_ndarray     : {t_convert:.2f} s  ({100 * t_convert / elapsed:.1f}%)")
    print(f"  copy to output : {t_copy:.2f} s  ({100 * t_copy / elapsed:.1f}%)")
    print(f"  frames decoded : {frames_decoded}  (targets={decoded}, duplicates={duplicate_frames})")
    print(f"  {fps:.1f} fps  |  {speed:.2f}x realtime")
    if args.verify:
        if verify_errors == 0:
            print("verify: all frames OK")
        else:
            print(f"verify: {verify_errors} errors found", file=sys.stderr)


def main() -> None:
    """CLI entry point — dispatches to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        description="Decode benchmarks for the sensor library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    sol_parser = subparsers.add_parser(
        "sol",
        help="Speed-of-light: raw PyAV decode, frames discarded immediately.",
        description=cmd_sol.__doc__,
    )
    _add_source_args(sol_parser)
    _add_threading_args(sol_parser)
    sol_parser.set_defaults(func=cmd_sol)

    kf_chunked_parser = subparsers.add_parser(
        "kf-chunked",
        help="Keyframe-chunked decode: seek+flush per GOP, all frames decoded and discarded.",
        description=cmd_kf_chunked.__doc__,
    )
    _add_source_args(kf_chunked_parser)
    _add_threading_args(kf_chunked_parser)
    kf_chunked_parser.set_defaults(func=cmd_kf_chunked)

    sensor_parser = subparsers.add_parser(
        "sensor",
        help="CameraSensor decode: exercises the full production path over a duration window.",
        description=cmd_sensor.__doc__,
    )
    _add_source_args(sensor_parser)
    _add_threading_args(sensor_parser, thread_type_default=None, thread_count_default=None)
    sensor_parser.add_argument(
        "--segment-duration",
        type=float,
        default=10.0,
        metavar="SECONDS",
        dest="segment_duration",
        help=(
            "Duration of each non-overlapping decode segment in seconds (default: 10.0). "
            "Maps directly to SamplingGrid stride_ns and duration_ns — the video is divided "
            "into back-to-back segments of this length and each is decoded independently."
        ),
    )
    sensor_parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "After each segment, independently re-decode every frame via raw PyAV seek+decode "
            "and compare pixel-by-pixel with the CameraSensor output. Not efficient — one seek "
            "per frame — but authoritative. Uses pts_stream from CameraData for exact seeks."
        ),
    )
    sensor_parser.set_defaults(func=cmd_sensor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
