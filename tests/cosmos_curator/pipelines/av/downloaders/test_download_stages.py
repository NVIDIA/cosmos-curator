"""Unit tests for AV downloader stages."""

import copy
import pathlib
from collections.abc import Callable
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest

from cosmos_curator.core.utils.config import operation_context
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.av.downloaders import download_stages
from cosmos_curator.pipelines.av.downloaders.download_stages import (
    ClipDownloader,
    SqliteDownloader,
    VideoDownloader,
)
from cosmos_curator.pipelines.av.utils import av_data_model
from cosmos_curator.pipelines.av.utils.av_data_info import CAMERA_MAPPING
from cosmos_curator.pipelines.av.utils.av_data_model import (
    AvClipAnnotationTask,
    AvSessionTrajectoryTask,
    AvSessionVideoSplitTask,
    ClipForAnnotation,
)
from cosmos_curator.pipelines.video.utils.decoder_utils import (
    VideoMetadata as DecoderVideoMetadata,
)


@pytest.fixture(autouse=True)
def fake_video_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide lightweight metadata extraction without invoking ffprobe."""

    def _fake_extract(_: bytes | str) -> DecoderVideoMetadata:
        return DecoderVideoMetadata(
            height=720,
            width=1280,
            fps=30.0,
            num_frames=300,
            video_codec="h264",
            pixel_format="yuv420p",
            video_duration=10.0,
            bit_rate_k=4_000,
            format_name="mp4",
            audio_codec=None,
        )

    monkeypatch.setattr(av_data_model, "extract_video_metadata", _fake_extract)


@pytest.fixture(autouse=True)
def pipeline_tmp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Ensure temporary files live in a writable directory during tests."""
    monkeypatch.setattr(operation_context, "get_tmp_dir", lambda: tmp_path)


@pytest.fixture
def fake_storage_client(monkeypatch: pytest.MonkeyPatch) -> object:
    """Patch storage client acquisition to keep interactions local."""
    fake_client = object()

    def _fake_get_storage_client(*, target_path: str) -> object:
        del target_path
        return fake_client

    monkeypatch.setattr(download_stages, "get_storage_client", _fake_get_storage_client)
    return fake_client


def test_process_data_collects_expected_videos_and_timestamps(
    fake_storage_client: object,
    monkeypatch: pytest.MonkeyPatch,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Download all cameras and attach parsed timestamp arrays."""
    downloader = VideoDownloader(
        output_prefix="s3://test-bucket/output",
        camera_format_id="L",
        prompt_variants=[],
    )
    downloader.stage_setup()

    task = split_task_factory(clip_counts=(), session_url="s3://test-bucket/session")

    files_in_session = [
        "clip-2-main.mp4",
        "clip-4-main.mp4",
        "clip-5-main.mp4",
        "clip-6-main.mp4",
        "clip-7-main.mp4",
        "clip-8-main.mp4",
        "timestamp_l1.csv",
        "readme.txt",
    ]
    monkeypatch.setattr(
        download_stages,
        "get_files_relative",
        lambda _session_url, client: files_in_session if client is fake_storage_client else [],
    )

    video_payloads = {camera_id: f"video-{camera_id}".encode() for camera_id in [2, 4, 5, 6, 7, 8]}
    timestamp_lines = "\n".join(f"{camera_id},{camera_id * 100.0}" for camera_id in [2, 4, 5, 6, 7, 8])

    def _fake_read_bytes(path: pathlib.Path | str, client: object) -> bytes:
        assert client is fake_storage_client
        name = str(path).split("/")[-1]
        if name.endswith(".csv"):
            return timestamp_lines.encode("utf-8")
        camera_id = int(name.split("-")[1])
        return video_payloads[camera_id]

    monkeypatch.setattr(download_stages, "read_bytes", _fake_read_bytes)

    result = downloader.process_data([task])
    assert result == [task]

    camera_ids = sorted(video.camera_id for video in task.videos)
    assert camera_ids == [2, 4, 5, 6, 7, 8]
    for video in task.videos:
        assert np.array_equal(video.encoded_data.resolve(), bytes_to_numpy(video_payloads[video.camera_id]))
        expected_timestamp = video.camera_id * 100
        assert video.timestamps_ms is not None
        assert np.array_equal(video.timestamps_ms, np.array([expected_timestamp], dtype=np.int64))
        assert video.metadata.height == 720


def test_process_data_continues_after_video_download_error(
    fake_storage_client: object,
    monkeypatch: pytest.MonkeyPatch,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Skip individual video failures but continue processing other cameras."""
    downloader = VideoDownloader(
        output_prefix="s3://test-bucket/output",
        camera_format_id="L",
        prompt_variants=[],
    )
    downloader.stage_setup()

    task = split_task_factory(clip_counts=(), session_url="s3://test-bucket/session")

    files = ["clip-2-main.mp4", "clip-4-main.mp4", "timestamp_l1.csv"]
    monkeypatch.setattr(
        download_stages,
        "get_files_relative",
        lambda _session_url, client: files if client is fake_storage_client else [],
    )

    def _fake_read_bytes(path: pathlib.Path | str, client: object) -> bytes:
        assert client is fake_storage_client
        name = str(path).split("/")[-1]
        if name.endswith(".csv"):
            return b"2,200.0\n4,400.0\n"
        camera_id = int(name.split("-")[1])
        if camera_id == 2:
            error_msg = "download failed"
            raise OSError(error_msg)
        return b"video-4"

    monkeypatch.setattr(download_stages, "read_bytes", _fake_read_bytes)

    result = downloader.process_data([task])
    assert result == []

    remaining_camera_ids = [video.camera_id for video in task.videos]
    assert remaining_camera_ids == [4]


def test_process_data_converts_h264_and_keeps_only_vri_camera(
    fake_storage_client: object,
    monkeypatch: pytest.MonkeyPatch,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Convert h264 sources and retain the VRI camera only."""
    downloader = VideoDownloader(
        output_prefix="s3://test-bucket/output",
        camera_format_id="L",
        prompt_variants=["vri"],
    )
    downloader.stage_setup()

    task = split_task_factory(clip_counts=(), session_url="s3://test-bucket/session")

    monkeypatch.setattr(
        download_stages,
        "get_files_relative",
        lambda *_: ["clip-2-main.h264", "timestamp_l1.csv"],
    )

    def _fake_read_bytes(path: pathlib.Path | str, client: object) -> bytes:
        assert client is fake_storage_client
        name = str(path).split("/")[-1]
        if name.endswith(".csv"):
            return b"2,1234.0\n"
        return b"h264-content"

    monkeypatch.setattr(download_stages, "read_bytes", _fake_read_bytes)

    captured_cmds: list[list[str]] = []

    def _fake_check_call(cmd: list[str], **_: object) -> None:
        captured_cmds.append(cmd)
        mp4_path = pathlib.Path(cmd[-1])
        mp4_path.write_bytes(b"converted-mp4")

    monkeypatch.setattr(
        "cosmos_curator.pipelines.av.downloaders.download_stages.subprocess.check_call",
        _fake_check_call,
    )

    result = downloader.process_data([task])
    assert result == [task]
    assert len(captured_cmds) == 1
    assert captured_cmds[0][:3] == ["ffmpeg", "-loglevel", "panic"]

    assert len(task.videos) == 1
    video = task.videos[0]
    assert video.camera_id == 2
    assert np.array_equal(video.encoded_data.resolve(), bytes_to_numpy(b"converted-mp4"))
    assert video.timestamps_ms is not None
    assert np.array_equal(video.timestamps_ms, np.array([1234], dtype=np.int64))


def test_process_data_drops_session_when_camera_missing(
    fake_storage_client: object,
    monkeypatch: pytest.MonkeyPatch,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Drop the session when expected camera data is incomplete."""
    downloader = VideoDownloader(
        output_prefix="s3://test-bucket/output",
        camera_format_id="L",
        prompt_variants=[],
    )
    downloader.stage_setup()

    task = split_task_factory(clip_counts=(), session_url="s3://test-bucket/session")

    present_cameras = [2, 4, 5, 6, 7]  # camera 8 is missing
    files = [f"clip-{camera}-main.mp4" for camera in present_cameras] + ["timestamp_l1.csv"]
    monkeypatch.setattr(download_stages, "get_files_relative", lambda *_: files)

    video_payloads = {camera_id: f"video-{camera_id}".encode() for camera_id in present_cameras}

    def _fake_read_bytes(path: pathlib.Path | str, client: object) -> bytes:
        assert client is fake_storage_client
        name = str(path).split("/")[-1]
        if name.endswith(".csv"):
            return "\n".join(f"{camera},{camera * 10.0}" for camera in present_cameras).encode("utf-8")
        camera_id = int(name.split("-")[1])
        return video_payloads[camera_id]

    monkeypatch.setattr(download_stages, "read_bytes", _fake_read_bytes)

    result = downloader.process_data([task])
    assert result == []


def test_process_data_drops_session_when_no_timestamps(
    fake_storage_client: object,
    monkeypatch: pytest.MonkeyPatch,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Drop the session when timestamps remain empty."""
    downloader = VideoDownloader(
        output_prefix="s3://test-bucket/output",
        camera_format_id="L",
        prompt_variants=[],
    )
    downloader.stage_setup()

    task = split_task_factory(clip_counts=(), session_url="s3://test-bucket/session")

    files = [
        "clip-2-main.mp4",
        "clip-4-main.mp4",
        "clip-5-main.mp4",
        "clip-6-main.mp4",
        "clip-7-main.mp4",
        "clip-8-main.mp4",
        "timestamp_l1.csv",
    ]
    monkeypatch.setattr(download_stages, "get_files_relative", lambda *_: files)

    payload = b"video"

    def _fake_read_bytes(path: pathlib.Path | str, client: object) -> bytes:
        assert client is fake_storage_client
        name = str(path).split("/")[-1]
        if name.endswith(".csv"):
            return b""
        return payload

    monkeypatch.setattr(download_stages, "read_bytes", _fake_read_bytes)

    result = downloader.process_data([task])
    assert result == []
    assert all(video.timestamps_ms is None or video.timestamps_ms.size == 0 for video in task.videos)


def test_process_data_skips_timestamp_read_when_extracting_from_video(
    fake_storage_client: object,
    monkeypatch: pytest.MonkeyPatch,
    split_task_factory: Callable[..., AvSessionVideoSplitTask],
) -> None:
    """Ensure timestamp files are ignored when extraction happens from video payloads."""
    downloader = VideoDownloader(
        output_prefix="s3://test-bucket/output",
        camera_format_id="L",
        prompt_variants=["vri"],
    )
    downloader.stage_setup()
    mapping_copy = copy.deepcopy(CAMERA_MAPPING["L"])
    mapping_copy["extract_timestamp_from_video"] = True
    downloader._camera_mapping_entry = mapping_copy
    downloader._extract_timestamp_from_video = True

    task = split_task_factory(clip_counts=(), session_url="s3://test-bucket/session")

    monkeypatch.setattr(
        download_stages,
        "get_files_relative",
        lambda _session_url, client: ["clip-2-main.mp4"] if client is fake_storage_client else [],
    )

    monkeypatch.setattr(download_stages, "read_bytes", lambda *_: b"video-2")

    read_timestamps_mock = Mock()
    monkeypatch.setattr(downloader, "_read_timestamps", read_timestamps_mock)

    result = downloader.process_data([task])
    assert result == []
    read_timestamps_mock.assert_not_called()
    assert len(task.videos) == 1
    assert task.videos[0].timestamps_ms is None


def test_sqlite_downloader_fetches_sqlite_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Download sqlite database blobs to the trajectory task."""
    downloader = SqliteDownloader(output_prefix="s3://test-bucket/output")
    fake_client = object()
    monkeypatch.setattr(
        "cosmos_curator.pipelines.av.downloaders.download_stages.s3_client.create_s3_client",
        lambda _prefix: fake_client,
    )

    task = AvSessionTrajectoryTask(session_url="s3://test-bucket/session")

    monkeypatch.setattr(
        download_stages,
        "get_files_relative",
        lambda _session_url, client: ["trajectory.sqlite", "notes.txt"] if client is fake_client else [],
    )
    monkeypatch.setattr(download_stages, "is_sqlite_file", lambda path: path.endswith(".sqlite"))
    monkeypatch.setattr(download_stages, "read_bytes", lambda *_: b"sqlite-data")

    downloader.stage_setup()
    result = downloader.process_data(task)
    assert result == [task]
    assert task.sqlite_db == b"sqlite-data"


def test_clip_downloader_populates_clip_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate encoded bytes for annotation clips."""
    downloader = ClipDownloader(output_prefix="s3://test-bucket/output")
    fake_client = object()
    monkeypatch.setattr(
        "cosmos_curator.pipelines.av.downloaders.download_stages.s3_client.create_s3_client",
        lambda _prefix: fake_client,
    )

    clips = [
        ClipForAnnotation(
            video_session_name="session",
            clip_session_uuid=uuid4(),
            uuid=uuid4(),
            camera_id=camera_id,
            span_index=index,
            url=f"s3://test-bucket/clip-{camera_id}.mp4",
        )
        for index, camera_id in enumerate([2, 4], start=1)
    ]
    task = AvClipAnnotationTask(clips=clips)

    def _fake_read_bytes(path: str | pathlib.Path, client: object) -> bytes:
        assert client is fake_client
        name = str(path).split("/")[-1]
        return f"bytes-{name}".encode()

    monkeypatch.setattr(download_stages, "read_bytes", _fake_read_bytes)

    downloader.stage_setup()
    result = downloader.process_data([task])
    assert result == [task]
    encoded_payloads = [clip.encoded_data.resolve() for clip in task.clips]
    expected = [bytes_to_numpy(b"bytes-clip-2.mp4"), bytes_to_numpy(b"bytes-clip-4.mp4")]
    assert all(np.array_equal(a, b) for a, b in zip(encoded_payloads, expected, strict=True))
