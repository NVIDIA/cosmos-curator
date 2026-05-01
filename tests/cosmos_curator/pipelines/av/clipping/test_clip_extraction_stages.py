# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Tests for clip extraction stages."""

import contextlib
import pathlib
import subprocess
from collections.abc import Callable, Generator
from typing import Any, cast

import numpy as np
import pytest
from loguru import logger

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.av.clipping import clip_extraction_stages as clip_module
from cosmos_curator.pipelines.av.clipping.clip_extraction_stages import (
    ClipTranscodingStage,
    FixedStrideExtractorStage,
)
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvSessionVideoSplitTask,
    AvVideo,
    ClipForTranscode,
)


def _stub_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    writer: Callable[[pathlib.Path], None] | None = None,
    *,
    exc: subprocess.CalledProcessError | None = None,
) -> None:
    """Replace subprocess.check_output in the module.

    When ``writer`` is provided, the stub materializes encoded clip data.
    When ``exc`` is provided, the stub raises it to emulate ffmpeg failure.
    """

    def _fake_check_output(
        command: list[str], *, cwd: pathlib.Path, stderr: int | None = None
    ) -> bytes:  # pragma: no cover - lambda syntax noise
        del command, stderr
        if exc is not None:
            raise exc
        if writer is not None:
            writer(pathlib.Path(cwd))
        return b""

    monkeypatch.setattr(cast("Any", clip_module).subprocess, "check_output", _fake_check_output)


def _ffmpeg_writer(clips: list[ClipForTranscode]) -> Callable[[pathlib.Path], None]:
    """Create a callable that materializes encoded data on disk for the provided clips."""

    def _writer(directory: pathlib.Path) -> None:
        for clip in clips:
            (directory / f"{clip.uuid}.mp4").write_bytes(f"encoded-{clip.span_index}".encode())

    return _writer


def test_extract_clips_populates_encoded_data(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clip_factory: Callable[..., ClipForTranscode],
) -> None:
    """Ensure `_extract_clips` writes output files and populates encoded data."""
    encoder_threads = 2
    stage = ClipTranscodingStage(
        encoder="libopenh264",
        openh264_bitrate=5,
        encoder_threads=encoder_threads,
        encode_batch_size=8,
    )
    clip_timestamps = np.arange(0, 160, 40, dtype=np.int64)
    clips = [
        clip_factory(span_index=0, timestamps_ms=clip_timestamps),
        clip_factory(span_index=1, timestamps_ms=clip_timestamps),
    ]
    (tmp_path / "input.mp4").write_bytes(b"fake-video")

    captured_commands: list[list[str]] = []

    def _fake_check_output(command: list[str], *, cwd: pathlib.Path, stderr: int | None = None) -> bytes:
        del stderr
        captured_commands.append(command)
        for clip in clips:
            (pathlib.Path(cwd) / f"{clip.uuid}.mp4").write_bytes(f"encoded-{clip.span_index}".encode())
        return b""

    monkeypatch.setattr(cast("Any", clip_module).subprocess, "check_output", _fake_check_output)

    stage._extract_clips(
        tmp_path,
        "input.mp4",
        force_pix_fmt=False,
        clips=clips,
        input_video="camera.mp4",
    )

    assert len(captured_commands) == 1
    ffmpeg_cmd = captured_commands[0]
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert "libopenh264" in ffmpeg_cmd
    assert "-b:v" in ffmpeg_cmd
    assert "5M" in ffmpeg_cmd
    assert ["-threads", str(encoder_threads)] in [ffmpeg_cmd[i : i + 2] for i in range(len(ffmpeg_cmd) - 1)]
    for clip in clips:
        assert f"{clip.uuid}.mp4" in ffmpeg_cmd
        assert np.array_equal(clip.encoded_data.resolve(), bytes_to_numpy(f"encoded-{clip.span_index}".encode()))


def test_extract_clips_nvenc_uses_hwaccel(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clip_factory: Callable[..., ClipForTranscode],
) -> None:
    """Assert NVENC path enables CUDA acceleration and quality safeguards."""
    stage = ClipTranscodingStage(
        encoder="h264_nvenc",
        encoder_threads=3,
        encode_batch_size=4,
        nb_streams_per_gpu=4,
        verbose=True,
    )
    clip_timestamps = np.arange(0, 160, 40, dtype=np.int64)
    clips = [clip_factory(span_index=0, timestamps_ms=clip_timestamps)]
    (tmp_path / "input.mp4").write_bytes(b"fake-video")

    captured_commands: list[list[str]] = []

    def _fake_check_output(command: list[str], *, cwd: pathlib.Path, stderr: int | None = None) -> bytes:
        del stderr
        captured_commands.append(command)
        for clip in clips:
            (pathlib.Path(cwd) / f"{clip.uuid}.mp4").write_bytes(b"encoded")
        return b""

    monkeypatch.setattr(cast("Any", clip_module).subprocess, "check_output", _fake_check_output)

    stage._extract_clips(
        tmp_path,
        "input.mp4",
        force_pix_fmt=True,
        clips=clips,
        input_video="camera.mp4",
    )

    assert pytest.approx(stage.num_gpus_per_worker) == 0.25
    assert stage.num_cpus_per_worker is None

    assert len(captured_commands) == 1
    ffmpeg_cmd = captured_commands[0]
    assert ["-hwaccel", "cuda"] in [ffmpeg_cmd[i : i + 2] for i in range(len(ffmpeg_cmd) - 1)]
    assert ["-hwaccel_output_format", "cuda"] in [ffmpeg_cmd[i : i + 2] for i in range(len(ffmpeg_cmd) - 1)]
    assert ["-pix_fmt", "yuv420p"] in [ffmpeg_cmd[i : i + 2] for i in range(len(ffmpeg_cmd) - 1)]
    assert "-rc:v" in ffmpeg_cmd
    assert "vbr" in ffmpeg_cmd
    assert np.array_equal(clips[0].encoded_data.resolve(), bytes_to_numpy(b"encoded"))


def test_extract_clips_handles_ffmpeg_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clip_factory: Callable[..., ClipForTranscode],
) -> None:
    """Ensure ffmpeg errors leave clip payloads untouched."""
    stage = ClipTranscodingStage(encoder="libopenh264", encode_batch_size=2)
    clip = clip_factory(span_index=0, payload=None, timestamps_ms=np.arange(0, 160, 40, dtype=np.int64))
    (tmp_path / "input.mp4").write_bytes(b"fake-video")

    error = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"], output=b"boom")
    _stub_ffmpeg(monkeypatch, exc=error)

    stage._extract_clips(
        tmp_path,
        "input.mp4",
        force_pix_fmt=False,
        clips=[clip],
        input_video="camera.mp4",
    )

    assert clip.encoded_data.resolve() is None
    generated = [p for p in tmp_path.glob("*.mp4") if p.name != "input.mp4"]
    assert generated == []


def test_process_data_clears_source_video(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clip_factory: Callable[..., ClipForTranscode],
    video_factory: Callable[..., AvVideo],
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Run `process_data` to confirm temp dir usage and clip hydration."""
    stage = ClipTranscodingStage(encoder="libopenh264", encode_batch_size=2)

    clip_timestamps = np.arange(0, 160, 40, dtype=np.int64)
    clips = [
        clip_factory(span_index=0, timestamps_ms=clip_timestamps),
        clip_factory(span_index=1, timestamps_ms=clip_timestamps),
    ]
    video = video_factory(
        num_clips=0,
        camera_id=2,
        encoded_data=bytes_to_numpy(b"video-bytes"),
        framerate=10.0,
        num_frames=10,
        height=1080,
        width=1920,
        size=1024,
        source_video="camera.mp4",
    )
    video.clips.extend(clips)
    task = split_task_factory(session_name="session", session_url="s3://session", videos=[video])

    @contextlib.contextmanager
    def _local_pipeline_dir(sub_dir: str | None = None) -> Generator[pathlib.Path, None, None]:
        target = tmp_path / "ray_pipeline"
        if sub_dir:
            target /= sub_dir
        target.mkdir(parents=True, exist_ok=True)
        yield target

    _stub_ffmpeg(monkeypatch, _ffmpeg_writer(clips))
    monkeypatch.setattr(clip_module, "make_pipeline_temporary_dir", _local_pipeline_dir)

    output_tasks = stage.process_data([task])
    assert output_tasks[0].encoder == "libopenh264"
    assert video.encoded_data.resolve() is None
    for clip in clips:
        assert np.array_equal(clip.encoded_data.resolve(), bytes_to_numpy(f"encoded-{clip.span_index}".encode()))


def test_fixed_stride_extractor_creates_aligned_clips(
    video_factory: Callable[..., AvVideo],
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Verify fixed-stride extractor builds synchronized clips and clears timestamps."""
    stage = FixedStrideExtractorStage(
        camera_format_id="U",
        clip_len_frames=4,
        clip_stride_frames=2,
        limit_clips=0,
    )

    video = video_factory(
        num_clips=0,
        camera_id=2,
        framerate=10.0,
        num_frames=10,
        encoded_data=bytes_to_numpy(b"video-bytes"),
        height=1080,
        width=1920,
        size=1024,
        source_video="camera.mp4",
    )
    task = split_task_factory(session_name="session", session_url="s3://session", videos=[video])

    output_tasks = stage.process_data([task])
    assert output_tasks is not None
    assert len(output_tasks) == 1

    clips = video.clips
    assert len(clips) == 3
    assert [clip.span_start for clip in clips] == [0.0, 0.2, 0.4]
    assert [clip.span_end for clip in clips] == [0.4, 0.6, 0.8]
    assert video.timestamps_ms is None


def test_fixed_stride_extractor_respects_limit_clips(
    video_factory: Callable[..., AvVideo],
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Ensure extraction stops after reaching the configured clip limit."""
    stage = FixedStrideExtractorStage(
        camera_format_id="U",
        clip_len_frames=4,
        clip_stride_frames=2,
        limit_clips=1,
    )

    video = video_factory(
        num_clips=0,
        camera_id=2,
        framerate=10.0,
        num_frames=10,
        encoded_data=bytes_to_numpy(b"video-bytes"),
        height=1080,
        width=1920,
        size=1024,
        source_video="camera.mp4",
    )
    task = split_task_factory(session_name="session", session_url="s3://session", videos=[video])

    stage.process_data([task])

    assert len(video.clips) == 1
    clip = video.clips[0]
    assert clip.span_index == 0
    assert clip.span_start == 0.0
    assert clip.span_end == pytest.approx(0.4)


def test_fixed_stride_extractor_skips_invalid_video(
    video_factory: Callable[..., AvVideo],
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Ensure extractor drops sessions when the video lacks encoded data."""
    stage = FixedStrideExtractorStage(camera_format_id="U")
    video = video_factory(num_clips=0, encoded_data=None)
    task = split_task_factory(session_name="session", session_url="s3://session", videos=[video])

    assert stage.process_data([task]) == []
    assert video.clips == []


def test_fixed_stride_extractor_skips_on_timestamp_verification_failure(
    video_factory: Callable[..., AvVideo],
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Reject clips when timestamp cadence deviates beyond tolerance."""
    stage = FixedStrideExtractorStage(
        camera_format_id="U",
        clip_len_frames=4,
        clip_stride_frames=4,
        limit_clips=1,
    )

    timestamps_ms = np.array([0, 100, 200, 500, 600, 700, 800, 900], dtype=np.int64)
    video = video_factory(
        num_clips=0,
        timestamps_ms=timestamps_ms,
        num_frames=8,
        framerate=10.0,
        encoded_data=bytes_to_numpy(b"video-bytes"),
    )
    task = split_task_factory(session_name="session", session_url="s3://session", videos=[video])

    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        stage.process_data([task])
    finally:
        logger.remove(sink_id)

    assert video.clips == []
    assert any("frame time variation" in message for message in messages)


def test_fixed_stride_extractor_handles_short_timestamp_list(
    video_factory: Callable[..., AvVideo],
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Stop extraction when the timestamps list is shorter than required."""
    stage = FixedStrideExtractorStage(
        camera_format_id="U",
        clip_len_frames=4,
        clip_stride_frames=2,
    )

    timestamps_ms = np.array([0, 100, 200], dtype=np.int64)
    video = video_factory(
        num_clips=0,
        timestamps_ms=timestamps_ms,
        num_frames=10,
        framerate=10.0,
        encoded_data=bytes_to_numpy(b"video-bytes"),
    )
    task = split_task_factory(session_name="session", session_url="s3://session", videos=[video])

    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        stage.process_data([task])
    finally:
        logger.remove(sink_id)

    assert video.clips == []
    assert any("end-frame-idx" in message for message in messages)
