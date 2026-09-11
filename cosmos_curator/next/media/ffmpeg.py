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

"""Typed FFprobe and FFmpeg contracts for fixed-stride video clips."""

import json
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from cosmos_curator.next.media.spans import Span, nanoseconds_to_ffmpeg_timestamp, seconds_to_nanoseconds

_DECODER_THREADS = 1
_FILTER_THREADS = 1
# libavutil/error.h: AVERROR_INVALIDDATA = FFERRTAG('I', 'N', 'D', 'A').
_AVERROR_INVALIDDATA = -1094995529


class TranscodeSettings(Protocol):
    """Structural settings required by the reusable FFmpeg transcoder.

    These are the settings that decide what the output media *is*, which is why
    callers may fold them into a clip identity. Scheduling knobs that only
    decide how fast it is produced -- thread counts, timeouts -- are passed
    separately so they stay out of that identity.
    """

    @property
    def video_encoder(self) -> str:
        """Return the FFmpeg video encoder name."""
        ...

    @property
    def video_bitrate(self) -> str:
        """Return the FFmpeg target video bitrate."""
        ...

    @property
    def audio_mode(self) -> str:
        """Return the optional audio stream handling mode."""
        ...


@dataclass(frozen=True)
class VideoMetadata:
    """Media fields needed by the v1 source and clip outcome contract."""

    duration_ns: int
    width: int
    height: int
    frame_rate: float
    frame_count: int | None
    video_codec: str


class MediaError(RuntimeError):
    """Base for media failures attributable to one source or clip.

    Callers distinguish these from every other exception: a ``MediaError`` is a
    data outcome for one item, while anything else is a broken environment or a
    programming error and should fail the task it happened in.
    """


class TranscodeError(MediaError):
    """Raised when FFmpeg cannot produce one planned clip."""


class ProbeError(RuntimeError):
    """FFprobe failed without establishing a permanent media problem.

    Timeouts, process termination, IO failures, and unrecognized responses may
    be retried. Exhausting retries must fail the task rather than turn this
    exception into a terminal media outcome.
    """


class InvalidMediaError(MediaError):
    """FFprobe explicitly reported invalid data in a complete local source.

    Callers must finish staging immutable input before probing a local path.
    The same diagnostic over HTTP remains a :class:`ProbeError`, since a
    transport failure can make otherwise valid media appear truncated.
    """


class UnsupportedMediaError(MediaError):
    """Raised when media is readable but does not satisfy the v1 contract.

    FFprobe returned metadata, but it cannot support the requested operation.
    Remote callers should verify this against a complete local copy before
    recording a terminal outcome.
    """


@lru_cache(maxsize=8)
def assert_video_encoder_available(encoder: str) -> None:
    """Fail fast when the requested FFmpeg encoder is absent.

    Memoized so workers can call this per task without re-spawning FFmpeg. The
    driver-side call only proves the driver's FFmpeg build; a heterogeneous
    cluster needs the same assertion where the transcoding actually happens.
    """
    command = ["ffmpeg", "-hide_banner", "-encoders"]
    try:
        result = subprocess.run(  # noqa: S603
            command, check=True, capture_output=True, timeout=30, stdin=subprocess.DEVNULL
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        msg = f"Failed to query FFmpeg encoders: {exc}"
        raise RuntimeError(msg) from exc
    output = result.stdout.decode("utf-8", errors="replace")
    if any(len(tokens := line.split(maxsplit=2)) > 1 and tokens[1] == encoder for line in output.splitlines()):
        return
    msg = f"FFmpeg does not expose required video encoder {encoder!r}"
    raise RuntimeError(msg)


def probe_video_bytes(video_bytes: bytes) -> VideoMetadata:
    """Probe the first video stream in an in-memory media object."""
    with tempfile.TemporaryDirectory(prefix="curator_next_probe_") as tmp_dir:
        path = Path(tmp_dir) / "media"
        path.write_bytes(video_bytes)
        return probe_video_path(path)


def probe_video_path(path: Path, *, timeout_s: int = 120) -> VideoMetadata:
    """Probe a worker-local media path."""
    return probe_video_source(path, timeout_s=timeout_s)


def probe_video_source(source: str | Path, *, timeout_s: int = 120) -> VideoMetadata:
    """Probe a complete local source or authenticated range-capable HTTP source."""
    source_text = str(source)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
    ]
    if _is_http_source(source_text):
        command.extend(("-seekable", "1"))
    command.append(source_text)
    try:
        result = subprocess.run(  # noqa: S603
            command, check=True, capture_output=True, timeout=timeout_s, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"FFprobe timed out after {exc.timeout}s on <source>"
        raise ProbeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        raise _probe_failure(exc, source_text) from exc
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        msg = "FFprobe returned invalid JSON"
        raise ProbeError(msg) from exc
    if not isinstance(payload, dict):
        msg = "FFprobe returned a non-object payload"
        raise ProbeError(msg)
    return _metadata_from_ffprobe(payload)


def _probe_failure(error: subprocess.CalledProcessError, source: str) -> ProbeError | InvalidMediaError:
    """Recognize local invalid data from FFprobe's structured error, never stderr."""
    diagnostic = _redacted_diagnostic(error.stderr, source)
    # A killed process is inconclusive even if it emitted an error before dying.
    if error.returncode > 0 and not _is_http_source(source):
        try:
            payload = json.loads(error.stdout or b"")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        else:
            detail = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(detail, dict) and detail.get("code") == _AVERROR_INVALIDDATA:
                return InvalidMediaError(diagnostic or "FFprobe reported invalid media data")
    return ProbeError(diagnostic or f"FFprobe exited with status {error.returncode}")


def transcode_span(
    source_bytes: bytes,
    span: Span,
    config: TranscodeSettings,
    *,
    encoder_threads: int = 1,
) -> bytes:
    """Transcode one logical span from bytes using the v1 media contract."""
    with tempfile.TemporaryDirectory(prefix="curator_next_transcode_") as tmp_dir:
        root = Path(tmp_dir)
        source_path = root / "source"
        clip_path = root / "clip.mp4"
        source_path.write_bytes(source_bytes)
        transcode_span_to_path(source_path, clip_path, span, config, encoder_threads=encoder_threads)
        return clip_path.read_bytes()


def transcode_span_to_path(  # noqa: PLR0913
    source: str | Path,
    destination: Path,
    span: Span,
    config: TranscodeSettings,
    *,
    encoder_threads: int = 1,
    timeout_s: int = 120,
) -> None:
    """Seek, transcode, and write one span without materializing source bytes in Python."""
    transcode_spans_to_paths(
        source,
        ((span, destination),),
        config,
        encoder_threads=encoder_threads,
        timeout_s=timeout_s,
    )


def transcode_spans_to_paths(
    source: str | Path,
    outputs: Sequence[tuple[Span, Path]],
    config: TranscodeSettings,
    *,
    encoder_threads: int = 1,
    timeout_s: int = 120,
) -> None:
    """Transcode several independently seekable spans in one FFmpeg process.

    Decoder and simple-filter parallelism stay single-threaded per clip because
    the batch already supplies clip-level concurrency. ``encoder_threads`` only
    controls each output encoder.
    """
    jobs = tuple(outputs)
    if not jobs:
        msg = "At least one span and destination are required"
        raise ValueError(msg)

    destinations = tuple(destination for _, destination in jobs)
    if len(set(destinations)) != len(destinations):
        msg = "FFmpeg output destinations must be unique"
        raise ValueError(msg)

    source_text = str(source)
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-filter_threads",
        str(_FILTER_THREADS),
        "-y",
    ]

    # Repeat the input so every output gets independent input-side seeking.
    # This is load-bearing for ranged cloud access and avoids decoding the
    # source from the beginning for every requested span.
    for span, _ in jobs:
        command.extend(("-threads", str(_DECODER_THREADS), "-ss", nanoseconds_to_ffmpeg_timestamp(span.start_ns)))
        if _is_http_source(source_text):
            command.extend(("-seekable", "1"))
        command.extend(("-i", source_text))

    for input_index, (span, destination) in enumerate(jobs):
        command.extend(
            (
                "-t",
                nanoseconds_to_ffmpeg_timestamp(span.duration_ns),
                "-map",
                f"{input_index}:v:0",
                "-c:v",
                config.video_encoder,
                "-b:v",
                config.video_bitrate,
                "-threads",
                str(encoder_threads),
                "-map",
                f"{input_index}:a:0?",
                "-c:a",
                config.audio_mode,
                "-movflags",
                "+faststart",
                str(destination),
            )
        )

    try:
        subprocess.run(  # noqa: S603
            command, check=True, capture_output=True, timeout=timeout_s, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired as exc:
        _remove_destinations(destinations)
        msg = f"FFmpeg timed out after {exc.timeout}s on <source>"
        raise TranscodeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        _remove_destinations(destinations)
        diagnostic = _redacted_diagnostic(exc.stderr, source_text)
        msg = diagnostic or f"FFmpeg exited with status {exc.returncode}"
        raise TranscodeError(msg) from exc

    missing_count = sum(not destination.is_file() for destination in destinations)
    if missing_count:
        _remove_destinations(destinations)
        msg = f"FFmpeg completed without creating {missing_count} of {len(destinations)} expected MP4s"
        raise TranscodeError(msg)


def _remove_destinations(destinations: Sequence[Path]) -> None:
    for destination in destinations:
        destination.unlink(missing_ok=True)


def _is_http_source(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _redacted_diagnostic(stderr: bytes | None, source: str) -> str:
    diagnostic = (stderr or b"").decode("utf-8", errors="replace").strip()
    return diagnostic.replace(source, "<source>")


def _metadata_from_ffprobe(payload: dict[str, Any]) -> VideoMetadata:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        msg = "FFprobe payload does not contain a streams list"
        raise ProbeError(msg)
    video_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        msg = "No video stream found"
        raise UnsupportedMediaError(msg)

    format_payload = payload.get("format")
    format_duration = format_payload.get("duration") if isinstance(format_payload, dict) else None
    raw_frame_count = video_stream.get("nb_frames")
    try:
        return VideoMetadata(
            duration_ns=_first_valid_duration_ns(video_stream.get("duration"), format_duration),
            width=int(video_stream["width"]),
            height=int(video_stream["height"]),
            frame_rate=_first_valid_frame_rate(video_stream.get("avg_frame_rate"), video_stream.get("r_frame_rate")),
            # Left unknown rather than estimated from duration and frame rate:
            # the published column is nullable precisely so a container that
            # does not carry a frame count says so instead of guessing.
            frame_count=int(raw_frame_count)
            if isinstance(raw_frame_count, str) and raw_frame_count.isdigit()
            else None,
            video_codec=str(video_stream["codec_name"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"FFprobe reported an unusable video stream: {exc}"
        raise UnsupportedMediaError(msg) from exc


def _first_valid_duration_ns(*values: Any) -> int:  # noqa: ANN401
    for value in values:
        if value is None:
            continue
        try:
            duration_ns = seconds_to_nanoseconds(str(value))
        except (ArithmeticError, ValueError):
            continue
        if duration_ns >= 0:
            return duration_ns
    msg = "FFprobe did not report a valid source duration"
    raise UnsupportedMediaError(msg)


def _first_valid_frame_rate(*values: Any) -> float:  # noqa: ANN401
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            rate = Fraction(value)
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return float(rate)
    msg = "FFprobe did not report a valid positive frame rate"
    raise UnsupportedMediaError(msg)
