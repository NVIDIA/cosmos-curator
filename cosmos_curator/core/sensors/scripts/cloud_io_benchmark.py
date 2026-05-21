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
r"""Cloud-IO benchmarks for the sensor library.

Measures wall time and bytes-read for ``make_index_and_metadata`` and
``CameraSensor.sample`` against local files, ``s3://`` URIs, and ``az://``
URIs. Validates that ``VideoIndexCreationMethod.FROM_HEADER`` stays bounded
on remote MP4s rather than scanning the entire object, and optionally
cross-validates remote results against a local reference.

Subcommands
-----------
index   Build a VideoIndex with FROM_HEADER and (optionally) FULL_DEMUX, report
        wall time, bytes returned to libav, seek count, and S3 GetObject
        request count + bytes-on-the-wire when applicable.

sample  Build a CameraSensor and run a small ``sample()`` workload at
        ``--target-fps`` over ``--duration-s``. Report the same IO metrics
        for the full lifecycle (index build + decode loop).

Both subcommands accept ``--reference-source <local-path>`` to assert
field-by-field equality of the resulting ``VideoIndex`` (and pixel-level
equality of decoded frames for ``sample``) against the local reference.

Run:
    python -m cosmos_curator.core.sensors.scripts.cloud_io_benchmark index \
        --source s3://bucket/clip.mp4 --s3-profile-name myprof \
        --reference-source /local/clip.mp4

    python -m cosmos_curator.core.sensors.scripts.cloud_io_benchmark sample \
        --source az://container/clip.mp4 --azure-profile-name myprof \
        --target-fps 1 --duration-s 10 \
        --reference-source /local/clip.mp4
"""

import argparse
import io
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast

import attrs
import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.data.video import VideoIndex
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.scripts._cli_cloud import (
    CloudCliError,
    add_cloud_credential_args,
    is_azure_uri,
    is_cloud_uri,
    is_s3_uri,
    make_azure_client,
    make_s3_client,
    open_cloud_source,
    validate_source,
)
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.types.types import DataSource, VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import make_index_and_metadata

# ---------------------------------------------------------------------------
# Byte-counting file-like proxy
# ---------------------------------------------------------------------------


class _CountingBinaryStream(io.BufferedIOBase):
    """Delegating wrapper around a binary stream that counts bytes read and seeks.

    Inherits from :class:`io.BufferedIOBase` so the sensor library's
    :func:`open_data_source` accepts it as a borrowed binary stream (the only
    file-like arm of :data:`DataSource` after the smart_open / URI removal).
    The library validates ``readable()`` + ``seekable()`` and uses the stream
    as-is, so all reads and seeks are counted here.
    """

    def __init__(self, inner: BinaryIO, stats: "IOStats") -> None:
        super().__init__()
        self._inner = inner
        self._stats = stats

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        try:
            return bool(self._inner.seekable())
        except (AttributeError, ValueError):
            return True

    def read(self, size: int | None = -1, /) -> bytes:
        data = self._inner.read(size if size is not None else -1)
        self._stats.read_calls += 1
        self._stats.bytes_read += len(data)
        if size is not None and size >= 0:
            self._stats.bytes_requested += size
        return data

    def read1(self, size: int = -1, /) -> bytes:
        read1 = getattr(self._inner, "read1", None)
        if read1 is None:
            return self.read(size)
        data: bytes = read1(size)
        self._stats.read_calls += 1
        self._stats.bytes_read += len(data)
        if size >= 0:
            self._stats.bytes_requested += size
        return data

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        new_pos = self._inner.seek(offset, whence)
        self._stats.seek_calls += 1
        self._stats.seek_positions.append(int(new_pos))
        return int(new_pos)

    def tell(self) -> int:
        return int(self._inner.tell())

    def close(self) -> None:
        if not self.closed:
            try:
                self._inner.close()
            finally:
                super().close()

    def __getattr__(self, name: str) -> object:
        # Fall back to the inner stream for attributes BufferedIOBase doesn't
        # expose (e.g. smart_open-specific name, mode).
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# IO statistics
# ---------------------------------------------------------------------------


@attrs.define
class IOStats:
    """Per-measurement IO accounting."""

    bytes_read: int = 0
    bytes_requested: int = 0
    read_calls: int = 0
    seek_calls: int = 0
    seek_positions: list[int] = attrs.field(factory=list)
    # S3-only: HTTP-level request count from the boto3 ``before-send.s3.GetObject`` hook.
    # We deliberately do NOT report bytes-on-the-wire from the Range header: smart_open
    # issues open-ended ranges (``bytes=START-``) and closes mid-stream, so the request
    # header doesn't carry the response size. ``bytes_read`` (above) is the source of
    # truth for bytes returned to libav, which equals bytes-on-the-wire modulo TCP
    # framing overhead.
    s3_request_count: int = 0

    def report(self) -> str:
        """Render a one-line summary."""
        seek_summary = (
            f"seeks={self.seek_calls} (positions={self.seek_positions[:5]}"
            f"{'...' if len(self.seek_positions) > 5 else ''})"  # noqa: PLR2004
        )
        s3_summary = f" | s3_requests={self.s3_request_count}" if self.s3_request_count > 0 else ""
        return (
            f"bytes_read={self.bytes_read:,} (requested={self.bytes_requested:,}) "
            f"reads={self.read_calls} {seek_summary}{s3_summary}"
        )


# ---------------------------------------------------------------------------
# Per-backend HTTP hooks
# ---------------------------------------------------------------------------


def _install_s3_hook(s3_client: Any, stats: IOStats) -> Any:  # noqa: ANN401  (boto3 client / event-handler types are dynamic)
    """Register a boto3 ``before-send.s3.GetObject`` hook that records the request count.

    Returns the registered callable so the caller can ``unregister`` it on exit.
    Bytes-on-the-wire are intentionally not derived from the request ``Range`` header
    because smart_open issues open-ended ranges (``bytes=START-``) and closes the
    connection mid-stream; the header doesn't carry the response size. Use
    ``IOStats.bytes_read`` for that.
    """
    events = s3_client.meta.events

    def _on_before_send(request: Any, **_kwargs: Any) -> None:  # noqa: ANN401, ARG001  (request kwarg name is fixed by boto3)
        stats.s3_request_count += 1

    events.register("before-send.s3.GetObject", _on_before_send)
    return _on_before_send


def _uninstall_s3_hook(s3_client: Any, hook: Any) -> None:  # noqa: ANN401
    s3_client.meta.events.unregister("before-send.s3.GetObject", hook)


# ---------------------------------------------------------------------------
# Measurement context manager
# ---------------------------------------------------------------------------


@contextmanager
def _open_measured_source(args: argparse.Namespace, stats: IOStats) -> Generator[DataSource, None, None]:
    """Open ``args.source`` for measurement and yield a counting ``DataSource``.

    For ``s3://`` sources: builds a boto3 S3 client, attaches the
    ``before-send.s3.GetObject`` hook to it, then hands the client into
    :func:`open_cloud_source` so smart_open reuses the same instrumented client.

    For ``az://`` sources: builds an Azure ``BlobServiceClient`` and hands it
    into :func:`open_cloud_source` (no HTTP-level hook is installed; bytes_read
    remains the source of truth for bytes on the wire).

    For local sources: opens the file directly.

    In every case the underlying stream is wrapped in a
    :class:`_CountingBinaryStream` so :class:`IOStats` accumulates ``read()`` /
    ``seek()`` activity at the file-like layer. The sensor library accepts
    the wrapped stream as an ``io.BufferedIOBase`` ``DataSource``.
    """
    source_str: str = args.source

    if is_s3_uri(source_str):
        s3_client = make_s3_client(source_str, args.s3_profile_name)
        hook = _install_s3_hook(s3_client, stats)
        try:
            with open_cloud_source(source_str, s3_client=s3_client) as raw:
                yield cast("DataSource", _CountingBinaryStream(raw, stats))
        finally:
            _uninstall_s3_hook(s3_client, hook)
    elif is_azure_uri(source_str):
        azure_client = make_azure_client(source_str, args.azure_profile_name)
        with open_cloud_source(source_str, azure_client=azure_client) as raw:
            yield cast("DataSource", _CountingBinaryStream(raw, stats))
    else:
        with Path(source_str).open("rb") as raw:
            yield cast("DataSource", _CountingBinaryStream(cast("BinaryIO", raw), stats))


# ---------------------------------------------------------------------------
# Reference (local) loaders
# ---------------------------------------------------------------------------


def _load_reference_index(reference_source: Path) -> VideoIndex:
    if not reference_source.is_file():
        msg = f"--reference-source is not a file: {reference_source}"
        raise CloudCliError(msg)
    index, _ = make_index_and_metadata(reference_source)
    return index


def _fail(msg: str) -> SystemExit:
    """Build a ``SystemExit`` with a uniform ``FAIL:`` prefix."""
    fail_msg = f"FAIL: {msg}"
    return SystemExit(fail_msg)


def _assert_index_equal(observed: VideoIndex, reference: VideoIndex, *, label: str) -> None:
    if observed != reference:
        msg = (
            f"VideoIndex mismatch ({label}): observed has {len(observed)} packets / "
            f"{len(observed.kf_pts_ns)} keyframes; reference has {len(reference)} packets / "
            f"{len(reference.kf_pts_ns)} keyframes; time_base "
            f"observed={observed.time_base} reference={reference.time_base}"
        )
        raise _fail(msg)


# ---------------------------------------------------------------------------
# index subcommand
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    """Measure ``make_index_and_metadata`` for FROM_HEADER and (optionally) FULL_DEMUX."""
    _validate_source_arg(args)
    print(f"source : {args.source}")

    methods: list[VideoIndexCreationMethod] = [VideoIndexCreationMethod.FROM_HEADER]
    if not args.skip_full_demux:
        methods.append(VideoIndexCreationMethod.FULL_DEMUX)

    reference_index: VideoIndex | None = None
    if args.reference_source is not None:
        ref_path = Path(args.reference_source)
        print(f"reference (local): {ref_path}")
        reference_index = _load_reference_index(ref_path)

    for method in methods:
        # `allow_header_fallback=False` for FROM_HEADER so we surface
        # _HeaderIndexUnavailableError rather than silently scanning the file.
        allow_fallback = method != VideoIndexCreationMethod.FROM_HEADER
        print(f"\n=== {method.name} ===")
        stats = IOStats()
        with _open_measured_source(args, stats) as data:
            t0 = time.perf_counter()
            index, metadata = make_index_and_metadata(
                data,
                index_method=method,
                allow_header_fallback=allow_fallback,
            )
            elapsed = time.perf_counter() - t0
        duration_s = (index.pts_ns[-1] - index.pts_ns[0]) / 1e9 if len(index) > 0 else 0.0
        print(
            f"  {len(index)} packets | {len(index.kf_pts_ns)} keyframes | "
            f"{duration_s:.2f}s video | codec={metadata.codec_name} {metadata.width}x{metadata.height}"
        )
        print(f"  wall_time={elapsed:.3f}s")
        print(f"  io: {stats.report()}")

        if reference_index is not None:
            _assert_index_equal(index, reference_index, label=method.name)
            print("  parity vs reference: OK")


# ---------------------------------------------------------------------------
# sample subcommand
# ---------------------------------------------------------------------------


def _build_sampling_grid(
    index: VideoIndex,
    *,
    target_fps: float,
    duration_s: float | None,
) -> SamplingGrid:
    """Build a uniform sampling grid at ``target_fps`` covering the first ``duration_s``."""
    start_ns = int(index.pts_ns[0])
    last_ns = int(index.pts_ns[-1])
    end_ns = last_ns + 1 if duration_s is None else min(start_ns + int(duration_s * 1_000_000_000), last_ns + 1)

    stride_ns = round(1_000_000_000 / target_fps)
    return SamplingGrid(
        start_ns=start_ns,
        exclusive_end_ns=end_ns,
        timestamps_ns=index.pts_ns,
        stride_ns=stride_ns,
        duration_ns=stride_ns,
    )


def _collect_sample(
    sensor: CameraSensor,
    spec: SamplingSpec,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Drive ``sensor.sample`` and return concatenated frames + sensor timestamps."""
    frames_chunks: list[npt.NDArray[np.uint8]] = []
    ts_chunks: list[npt.NDArray[np.int64]] = []
    for batch in sensor.sample(spec):
        if len(batch.frames) == 0:
            continue
        frames_chunks.append(batch.frames)
        ts_chunks.append(batch.sensor_timestamps_ns)
    if not frames_chunks:
        return np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.int64)
    return np.concatenate(frames_chunks, axis=0), np.concatenate(ts_chunks, axis=0)


def cmd_sample(args: argparse.Namespace) -> None:
    """Measure a small ``CameraSensor.sample`` workload end-to-end."""
    _validate_source_arg(args)
    print(f"source : {args.source}")
    print(f"target_fps={args.target_fps} duration_s={args.duration_s}")

    reference_frames: npt.NDArray[np.uint8] | None = None
    reference_ts: npt.NDArray[np.int64] | None = None
    if args.reference_source is not None:
        ref_path = Path(args.reference_source)
        if not ref_path.is_file():
            msg = f"--reference-source is not a file: {ref_path}"
            raise CloudCliError(msg)
        print(f"reference (local): {ref_path}")
        ref_sensor = CameraSensor(ref_path)
        ref_grid = _build_sampling_grid(ref_sensor.video_index, target_fps=args.target_fps, duration_s=args.duration_s)
        reference_frames, reference_ts = _collect_sample(ref_sensor, SamplingSpec(grid=ref_grid))

    print("\n=== CameraSensor lifecycle ===")
    stats = IOStats()
    # The CameraSensor's lifecycle (index build + .sample()) shares a single
    # underlying BinaryIO after the refactor — PyAV seeks within it for both
    # phases. This is a behavioural change vs. the prior two-open lifecycle
    # (one smart_open per phase); HTTP GetObject counts may compact.
    with _open_measured_source(args, stats) as data:
        t0 = time.perf_counter()
        sensor = CameraSensor(data)
        t_index = time.perf_counter() - t0

        grid = _build_sampling_grid(sensor.video_index, target_fps=args.target_fps, duration_s=args.duration_s)
        spec = SamplingSpec(grid=grid)
        n_windows = sum(1 for _ in grid)

        t1 = time.perf_counter()
        frames, sensor_ts = _collect_sample(sensor, spec)
        t_sample = time.perf_counter() - t1
        elapsed = time.perf_counter() - t0

    print(
        f"  {n_windows} windows | sampled {len(frames)} frames | "
        f"frame_shape={frames.shape if len(frames) else '(empty)'}"
    )
    print(f"  wall_time={elapsed:.3f}s  (index_build={t_index:.3f}s  sample={t_sample:.3f}s)")
    print(f"  io: {stats.report()}")

    if reference_frames is not None:
        if reference_frames.shape != frames.shape:
            msg = f"frame shape mismatch: observed={frames.shape} reference={reference_frames.shape}"
            raise _fail(msg)
        if not np.array_equal(frames, reference_frames):
            diff = np.abs(frames.astype(np.int32) - reference_frames.astype(np.int32))
            msg = f"frame pixel mismatch: max_diff={int(diff.max())} mean_diff={float(diff.mean()):.3f}"
            raise _fail(msg)
        if reference_ts is not None and not np.array_equal(sensor_ts, reference_ts):
            msg = (
                f"sensor_timestamps_ns mismatch: "
                f"observed[:3]={sensor_ts[:3].tolist()} reference[:3]={reference_ts[:3].tolist()}"
            )
            raise _fail(msg)
        print("  parity vs reference: OK (frames + timestamps)")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_source_arg(args: argparse.Namespace) -> None:
    """Validate ``args.source`` and exit cleanly on credential / path errors."""
    try:
        validate_source(args.source)
        # Pre-flight credential resolution so failures surface before measurement.
        if is_cloud_uri(args.source) and not is_s3_uri(args.source) and not is_azure_uri(args.source):
            msg = f"unsupported cloud URI: {args.source!r}"
            raise CloudCliError(msg)  # noqa: TRY301
    except CloudCliError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(2)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _add_common_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        required=True,
        help="Video source: local file path, or s3:// / az:// URI.",
    )
    parser.add_argument(
        "--reference-source",
        default=None,
        help="Optional local file used as a golden reference for parity checks.",
    )
    add_cloud_credential_args(parser)


def main(argv: list[str] | None = None) -> int:
    """Run the cloud-IO benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Measure wall time and bytes-read for sensor-library cloud sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    index_parser = subparsers.add_parser(
        "index",
        help="Measure make_index_and_metadata (FROM_HEADER and FULL_DEMUX).",
        description=cmd_index.__doc__,
    )
    _add_common_source_args(index_parser)
    index_parser.add_argument(
        "--skip-full-demux",
        action="store_true",
        help="Skip the FULL_DEMUX measurement (useful when the object is large).",
    )
    index_parser.set_defaults(func=cmd_index)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Measure a CameraSensor.sample() workload.",
        description=cmd_sample.__doc__,
    )
    _add_common_source_args(sample_parser)
    sample_parser.add_argument(
        "--target-fps",
        type=float,
        default=1.0,
        help="Sampling rate in Hz (default: 1.0).",
    )
    sample_parser.add_argument(
        "--duration-s",
        type=float,
        default=10.0,
        help="Limit sampling to the first DURATION_S seconds (default: 10.0; pass <=0 for full clip).",
    )
    sample_parser.set_defaults(func=cmd_sample)

    args = parser.parse_args(argv)
    if getattr(args, "duration_s", None) is not None and args.duration_s <= 0:
        args.duration_s = None
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
