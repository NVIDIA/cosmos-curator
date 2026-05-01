# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for VideoDownloader inline remux behaviour."""

# ruff: noqa: ARG002  # mock args required by @patch.object decorator order, not referenced directly

import pathlib
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.read_write.download_stages import VideoDownloader
from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask, Video


def _make_downloader() -> VideoDownloader:
    return VideoDownloader(input_path="/fake", input_s3_profile_name="default")


def _make_task(video: Video) -> Mock:
    task = Mock(spec=SplitPipeTask)
    task.videos = [video]
    task.get_major_size.return_value = 0
    task.stage_perf = {}
    return task


class TestVideoDownloaderRemux:
    """Inline remux behaviour in VideoDownloader.process_data."""

    @patch.object(Video, "populate_timestamps")
    @patch("cosmos_curator.pipelines.video.read_write.download_stages.remux_if_needed")
    @patch.object(VideoDownloader, "_extract_and_validate_metadata", return_value=True)
    @patch.object(VideoDownloader, "_download_video_bytes", return_value=True)
    def test_mp4_passthrough(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_remux: MagicMock,
        mock_populate: MagicMock,
    ) -> None:
        """mp4: remux_if_needed is called, returns False, was_remuxed stays False."""
        mock_remux.return_value = False
        video = Video(input_video=pathlib.Path("test.mp4"))
        task = _make_task(video)

        result = _make_downloader().process_data([task])

        assert result == [task]
        mock_remux.assert_called_once_with(video, threads=1)
        assert video.was_remuxed is False
        assert video.errors == {}

    @patch.object(Video, "populate_timestamps")
    @patch("cosmos_curator.pipelines.video.read_write.download_stages.remux_if_needed")
    @patch.object(VideoDownloader, "_extract_and_validate_metadata", return_value=True)
    @patch.object(VideoDownloader, "_download_video_bytes", return_value=True)
    def test_mpegts_remux_inline(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_remux: MagicMock,
        mock_populate: MagicMock,
    ) -> None:
        """mpegts: remux_if_needed is called, returns True, was_remuxed set True."""
        mock_remux.return_value = True
        video = Video(input_video=pathlib.Path("test.ts"))
        task = _make_task(video)

        result = _make_downloader().process_data([task])

        assert result == [task]
        mock_remux.assert_called_once_with(video, threads=1)
        assert video.was_remuxed is True
        assert video.errors == {}

    @patch("cosmos_curator.pipelines.video.read_write.download_stages.remux_if_needed")
    @patch.object(VideoDownloader, "_log_video_info")
    @patch.object(VideoDownloader, "_extract_and_validate_metadata", return_value=True)
    @patch.object(VideoDownloader, "_download_video_bytes", return_value=True)
    def test_remux_failure_isolated(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_log: MagicMock,
        mock_remux: MagicMock,
    ) -> None:
        """Remux exception: error recorded, was_remuxed stays False, _log_video_info not called."""
        mock_remux.side_effect = RuntimeError("ffmpeg exploded")
        video = Video(input_video=pathlib.Path("test.ts"))
        task = _make_task(video)

        result = _make_downloader().process_data([task])

        assert result == [task]
        assert video.errors.get("remux") == "ffmpeg exploded"
        assert video.was_remuxed is False
        mock_log.assert_not_called()

    @patch("cosmos_curator.pipelines.video.read_write.download_stages.remux_if_needed")
    @patch.object(VideoDownloader, "_download_video_bytes", return_value=False)
    def test_download_failure_skips_remux(
        self,
        mock_download: MagicMock,
        mock_remux: MagicMock,
    ) -> None:
        """Download failure: remux_if_needed never called, was_remuxed stays False."""
        video = Video(input_video=pathlib.Path("test.mp4"))
        task = _make_task(video)

        result = _make_downloader().process_data([task])

        assert result == [task]
        mock_remux.assert_not_called()
        assert video.was_remuxed is False

    @patch.object(VideoDownloader, "_log_video_info")
    @patch("cosmos_curator.pipelines.video.read_write.download_stages.remux_if_needed", return_value=False)
    @patch.object(VideoDownloader, "_extract_and_validate_metadata", return_value=True)
    @patch.object(VideoDownloader, "_download_video_bytes", return_value=True)
    def test_timestamps_populated_after_download(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_remux: MagicMock,
        mock_log: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """populate_timestamps() is called after successful remux on every video."""
        expected = np.array([0.0, 0.033], dtype=np.float32)
        monkeypatch.setattr(
            "cosmos_curator.pipelines.video.utils.data_model.get_video_timestamps",
            lambda _data: expected,
        )
        video = Video(input_video=pathlib.Path("test.mp4"))
        video.encoded_data = bytes_to_numpy(b"fake")
        task = _make_task(video)

        _make_downloader().process_data([task])

        assert video.timestamps is not None
        assert np.array_equal(video.timestamps, expected)

    @patch.object(VideoDownloader, "_log_video_info")
    @patch("cosmos_curator.pipelines.video.read_write.download_stages.remux_if_needed", return_value=False)
    @patch.object(VideoDownloader, "_extract_and_validate_metadata", return_value=True)
    @patch.object(VideoDownloader, "_download_video_bytes", return_value=True)
    def test_timestamps_failure_does_not_drop_video(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_remux: MagicMock,
        mock_log: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """populate_timestamps() failure records error but video is still returned in task."""
        monkeypatch.setattr(
            "cosmos_curator.pipelines.video.utils.data_model.get_video_timestamps",
            lambda _data: (_ for _ in ()).throw(RuntimeError("decoder failed")),
        )
        video = Video(input_video=pathlib.Path("test.mp4"))
        video.encoded_data = bytes_to_numpy(b"fake")
        task = _make_task(video)

        result = _make_downloader().process_data([task])

        assert result == [task]
        assert "timestamps" in video.errors
        assert "decoder failed" in video.errors["timestamps"]
        assert video.timestamps is None

    @patch("cosmos_curator.pipelines.video.read_write.download_stages.remux_if_needed")
    @patch.object(VideoDownloader, "_log_video_info")
    @patch.object(VideoDownloader, "_extract_and_validate_metadata", return_value=True)
    @patch.object(VideoDownloader, "_download_video_bytes", return_value=True)
    def test_remux_failure_skips_populate_timestamps(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_log: MagicMock,
        mock_remux: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Remux exception: populate_timestamps() never called, no timestamps error recorded."""
        mock_remux.side_effect = RuntimeError("ffmpeg exploded")
        populate_calls: list[str] = []
        monkeypatch.setattr(
            "cosmos_curator.pipelines.video.utils.data_model.get_video_timestamps",
            lambda _data: populate_calls.append("called"),
        )
        video = Video(input_video=pathlib.Path("test.ts"))
        task = _make_task(video)

        _make_downloader().process_data([task])

        assert len(populate_calls) == 0
        assert "timestamps" not in video.errors
