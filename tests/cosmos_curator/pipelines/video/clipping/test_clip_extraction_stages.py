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
"""Tests for clip_extraction_stages (chunking and slice_video_clips)."""

import pathlib
import uuid
from contextlib import AbstractContextManager, nullcontext
from typing import Any
from unittest.mock import patch

import pytest

from cosmos_curator.pipelines.video.clipping.clip_extraction_stages import ClipTranscodingStage, slice_video_clips
from cosmos_curator.pipelines.video.utils.data_model import Clip, Video, VideoMetadata


def _make_video_for_slice(num_clips: int, clip_chunk_index: int = 0) -> Video:
    """Build a Video with num_clips dummy clips for slice_video_clips tests."""
    clips = [
        Clip(
            uuid=uuid.uuid4(),
            source_video="test.mp4",
            span=(float(i * 10), float((i + 1) * 10)),
        )
        for i in range(num_clips)
    ]
    return Video(
        input_video=pathlib.Path("test.mp4"),
        relative_path="test.mp4",
        metadata=VideoMetadata(duration=float(num_clips * 10), size=1000),
        clips=clips,
        num_total_clips=num_clips,
        num_clip_chunks=2,
        clip_chunk_index=clip_chunk_index,
        errors={"stage": "err"},
    )


@pytest.mark.parametrize(
    (
        "num_clips",
        "start",
        "end",
        "chunk_index",
        "num_chunks",
        "expected_num_clips",
        "expected_chunk_index",
        "source_clip_chunk_index",
        "raises",
    ),
    [
        # Success: sub set
        (5, 1, 4, 0, 2, 3, 0, 0, nullcontext()),
        # Success: explicit chunk_index
        (5, 0, 2, 1, 2, 2, 1, 0, nullcontext()),
        # Success: chunk_index from video
        (5, 0, 2, 2, 2, 2, 2, 2, nullcontext()),
        # Success: full range
        (4, 0, 4, 0, 2, 4, 0, 0, nullcontext()),
        # Success: single clip
        (5, 2, 3, 0, 2, 1, 0, 0, nullcontext()),
        # Failure: end < start  # noqa: ERA001
        (5, 3, 2, 0, 2, None, None, 0, pytest.raises(ValueError, match="End index 2 is less than start index 3")),
        # Failure: start < 0  # noqa: ERA001
        (5, -1, 2, 0, 2, None, None, 0, pytest.raises(ValueError, match="out of range")),
        # Failure: end > len(clips)  # noqa: ERA001
        (5, 0, 6, 0, 2, None, None, 0, pytest.raises(ValueError, match="out of range")),
    ],
)
def test_slice_video_clips(  # noqa: PLR0913
    num_clips: int,
    start: int,
    end: int,
    chunk_index: int,
    num_chunks: int,
    expected_num_clips: int | None,
    expected_chunk_index: int | None,
    source_clip_chunk_index: int,
    raises: AbstractContextManager[Any],
) -> None:
    """Test slice_video_clips: valid slices return new Video with correct clips/chunk_index; invalid args raise."""
    video = _make_video_for_slice(num_clips, clip_chunk_index=source_clip_chunk_index)
    with raises:
        result = slice_video_clips(video, start, end, chunk_index, num_chunks)
        assert result is not video
        assert expected_num_clips is not None
        assert expected_chunk_index is not None
        assert len(result.clips) == expected_num_clips
        assert result.clip_chunk_index == expected_chunk_index
        assert result.num_clip_chunks == num_chunks
        assert result.input_video == video.input_video
        assert result.relative_path == video.relative_path
        assert result.num_total_clips == num_clips
        if expected_num_clips > 0:
            assert result.clips[0] is video.clips[start]
        result.errors["other"] = "new"
        assert "other" not in video.errors
        assert result.errors is not video.errors


class TestMaxOutputFrames:
    """Tests for the max_output_frames FPS-limiting feature in ClipTranscodingStage."""

    @staticmethod
    def _build_clips_and_stage(
        spans: list[tuple[float, float]],
        max_output_frames: int | None,
    ) -> tuple[ClipTranscodingStage, list[Clip]]:
        clips = [Clip(uuid=uuid.uuid4(), source_video="test.mp4", span=span) for span in spans]
        stage = ClipTranscodingStage(max_output_frames=max_output_frames)
        return stage, clips

    @staticmethod
    def _extract_fps_limit_flags(command: list[str], clips: list[Clip]) -> list[tuple[str | None, str | None]]:
        """Extract the -r and -frames:v values for each clip output in an ffmpeg command.

        Returns a list of (r_value, frames_v_value) tuples, one per clip output.
        """
        output_filenames = {f"{clip.uuid}.mp4" for clip in clips}
        results: list[tuple[str | None, str | None]] = []
        i = 0
        while i < len(command):
            if command[i] in output_filenames:
                r_val = None
                frames_val = None
                for j in range(i - 1, -1, -1):
                    if command[j] == "-r" and j + 1 < len(command):
                        r_val = command[j + 1]
                    if command[j] == "-frames:v" and j + 1 < len(command):
                        frames_val = command[j + 1]
                    if command[j] in output_filenames:
                        break  # hit previous output
                results.append((r_val, frames_val))
            i += 1
        return results

    @pytest.mark.parametrize(
        ("source_fps", "span", "max_frames", "expect_r_flag"),
        [
            # 60fps * 30s = 1800 frames > 186 → needs limiting
            pytest.param(60.0, (0.0, 30.0), 186, True, id="high_fps_long_clip"),
            # 10fps * 5s = 50 frames ≤ 186 → no limiting
            pytest.param(10.0, (0.0, 5.0), 186, False, id="low_fps_short_clip"),
            # 2fps * 30s = 60 frames ≤ 186 → no limiting (never upscale)
            pytest.param(2.0, (0.0, 30.0), 186, False, id="very_low_fps"),
            # max_output_frames disabled → no limiting
            pytest.param(60.0, (0.0, 30.0), None, False, id="disabled"),
            # source_fps unknown → no limiting
            pytest.param(None, (0.0, 30.0), 186, False, id="unknown_fps"),
        ],
    )
    def test_r_flag_presence(
        self,
        source_fps: float | None,
        span: tuple[float, float],
        max_frames: int | None,
        *,
        expect_r_flag: bool,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify that -r flag is added only when frame count exceeds the limit."""
        stage, clips = self._build_clips_and_stage([span], max_frames)

        captured_cmd: list[str] = []

        def fake_check_output(cmd: list[str], **_kwargs: object) -> bytes:
            captured_cmd.extend(cmd)
            # Create a dummy output file so the stage doesn't fail
            for clip in clips:
                (tmp_path / f"{clip.uuid}.mp4").write_bytes(b"\x00" * 100)
            return b""

        with patch("subprocess.check_output", side_effect=fake_check_output):
            stage._extract_clips(
                tmp_path,
                "input.mp4",
                force_pix_fmt=False,
                use_bit_rate=None,
                clips=clips,
                input_video="test.mp4",
                source_fps=source_fps,
            )

        flags = self._extract_fps_limit_flags(captured_cmd, clips)
        assert len(flags) == 1
        r_val, frames_val = flags[0]
        if expect_r_flag:
            assert r_val is not None
            assert frames_val is not None
            duration = span[1] - span[0]
            assert max_frames is not None
            expected_fps = max_frames / duration
            assert abs(float(r_val) - expected_fps) < 0.01
            assert frames_val == str(max_frames)
        else:
            assert r_val is None
            assert frames_val is None

    def test_mixed_clips_selective_limiting(self, tmp_path: pathlib.Path) -> None:
        """When multiple clips are batched, only those exceeding the limit get -r."""
        spans = [(0.0, 30.0), (0.0, 2.0)]  # 30s clip needs limiting at 60fps, 2s doesn't
        stage, clips = self._build_clips_and_stage(spans, max_output_frames=186)

        captured_cmd: list[str] = []

        def fake_check_output(cmd: list[str], **_kwargs: object) -> bytes:
            captured_cmd.extend(cmd)
            for clip in clips:
                (tmp_path / f"{clip.uuid}.mp4").write_bytes(b"\x00" * 100)
            return b""

        with patch("subprocess.check_output", side_effect=fake_check_output):
            stage._extract_clips(
                tmp_path,
                "input.mp4",
                force_pix_fmt=False,
                use_bit_rate=None,
                clips=clips,
                input_video="test.mp4",
                source_fps=60.0,
            )

        flags = self._extract_fps_limit_flags(captured_cmd, clips)
        assert len(flags) == 2
        # 60fps * 30s = 1800 > 186 → limited
        assert flags[0][0] is not None
        assert abs(float(flags[0][0]) - 186.0 / 30.0) < 0.01
        assert flags[0][1] == "186"
        # 60fps * 2s = 120 < 186 → not limited
        assert flags[1][0] is None
        assert flags[1][1] is None
