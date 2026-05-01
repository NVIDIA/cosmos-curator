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
"""Tests for the clip metadata writer stage."""

import base64
import json
import pickle
import uuid
from pathlib import Path
from types import SimpleNamespace

import lance
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.storage import storage_client, storage_utils
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import (
    ClipWriterStage,
    _archive_processed_sidecars,
    consolidate_lance_fragments,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, VideoMetadata, Window


def _create_stage(output_dir: Path, input_dir: Path, **overrides: object) -> ClipWriterStage:
    """Instantiate the stage with default options and optional overrides."""
    params = {
        "output_path": str(output_dir),
        "input_path": str(input_dir),
        "output_s3_profile_name": "default",
        "upload_clips": True,
        "upload_clip_info_in_chunks": False,
        "upload_clip_info_in_lance": False,
        "upload_cds_parquet": True,
        "dry_run": False,
        "generate_embeddings": True,
        "embedding_algorithm": "internvideo2",
        "embedding_model_version": "v1",
        "generate_previews": False,
        "caption_models": ["qwen"],
        "enhanced_caption_models": ["qwen_plus"],
        "generate_cosmos_predict_dataset": "disable",
        "verbose": False,
        "log_stats": False,
    }
    params.update(overrides)
    stage = ClipWriterStage(**params)
    stage.stage_setup()
    return stage


class _FailingDeleteClient(storage_client.StorageClient):
    """Storage client stub that raises on delete to exercise cleanup errors."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})

    def object_exists(self, dest: storage_client.StoragePrefix) -> bool:
        return str(dest) in self.objects

    def upload_bytes(self, dest: storage_client.StoragePrefix, data: bytes) -> None:
        self.objects[str(dest)] = data

    def upload_bytes_uri(self, uri: str, data: bytes, _chunk_size_bytes: int = 100) -> None:
        self.objects[uri] = data

    def download_object_as_bytes(self, uri: storage_client.StoragePrefix, _chunk_size_bytes: int = 10) -> bytes:
        try:
            return self.objects[str(uri)]
        except KeyError as exc:
            raise FileNotFoundError(str(uri)) from exc

    def download_objects_as_bytes(self, uris: list[storage_client.StoragePrefix]) -> list[bytes]:
        return [self.download_object_as_bytes(uri) for uri in uris]

    def list_recursive_directory(
        self, uri: storage_client.StoragePrefix, limit: int = 0
    ) -> list[storage_client.StoragePrefix]:
        _ = limit
        prefix = str(uri).rstrip("/") + "/"
        return [storage_utils.path_to_prefix(path) for path in sorted(self.objects) if path.startswith(prefix)]

    def list_recursive(
        self, prefix: storage_client.StoragePrefix, limit: int = 0
    ) -> list[dict[str, object]]:  # pragma: no cover - unused
        raise NotImplementedError

    def upload_file(
        self,
        local_path: str,
        remote_path: storage_client.StoragePrefix,
        _chunk_size: int = 100,
    ) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def sync_remote_to_local(
        self,
        remote_prefix: storage_client.StoragePrefix,
        local_dir: Path,
        *,
        delete: bool = False,
        _chunk_size_bytes: int = 10,
    ) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def make_background_uploader(
        self,
        _chunk_size_bytes: int = 100,
    ) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def delete_object(self, dest: storage_client.StoragePrefix) -> None:
        error_msg = f"delete failed for {dest}"
        raise RuntimeError(error_msg)


@pytest.fixture(autouse=True)
def fake_extract_video_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid invoking ffprobe in tests."""

    def _fake_extract_video_metadata(_: bytes) -> SimpleNamespace:
        return SimpleNamespace(
            width=1920,
            height=1080,
            fps=30.0,
            num_frames=60,
            video_codec="h264",
        )

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.utils.data_model.extract_video_metadata",
        _fake_extract_video_metadata,
    )


def _build_video(
    video_path: Path,
    clip: Clip,
    *,
    clip_chunk_index: int = 0,
    relative_path: str = "",
) -> Video:
    """Assemble Video metadata wrapper used by the stage."""
    metadata = VideoMetadata(
        height=1080,
        width=1920,
        framerate=30.0,
        num_frames=60,
        duration=2.0,
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac",
    )
    return Video(
        input_video=video_path,
        metadata=metadata,
        clips=[clip],
        filtered_clips=[],
        num_total_clips=1,
        num_clip_chunks=1,
        clip_chunk_index=clip_chunk_index,
        relative_path=relative_path,
    )


def _stage_with_main_clip(tmp_path: Path) -> tuple[ClipWriterStage, SplitPipeTask, Clip, Window, Path]:
    """Create a standard stage/task/clip tuple for integration testing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input-video")

    stage = _create_stage(output_dir, input_dir)

    main_window = Window(
        start_frame=0,
        end_frame=30,
        caption={"qwen": "main caption"},
        caption_status="success",
        enhanced_caption={"qwen_plus": "enhanced view"},
        webp_bytes=b"webp-content",
        t5_xxl_embedding={"default": np.array([1, 2, 3], dtype=np.int32)},
    )
    filtered_window = Window(
        start_frame=0,
        end_frame=30,
        caption={"qwen_rejection_reasons": "too blurry"},
    )
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=bytes_to_numpy(b"clip-bytes"),
        windows=[main_window],
        filter_windows=[filtered_window],
    )
    clip.intern_video_2_embedding = np.array([0.1, 0.2], dtype=np.float32)

    video = _build_video(video_path, clip)
    task = SplitPipeTask(session_id="test-session", video=video)
    return stage, task, clip, main_window, output_dir


def _assert_payloads_cleared(clip: Clip, window: Window) -> None:
    """Ensure transient buffers are released after processing."""
    assert clip.encoded_data.resolve() is None
    assert clip.intern_video_2_embedding is None
    assert window.webp_bytes.resolve() is None
    assert window.caption == {}
    assert window.enhanced_caption == {}


def _read_json(path: Path) -> dict[str, object]:
    """Load JSON data from disk."""
    return json.loads(path.read_text())


def _assert_embeddings_written(output_dir: Path, clip: Clip, video_uuid: uuid.UUID) -> None:
    """Validate clip-level and chunk-level embedding outputs."""
    embedding_pickle_path = output_dir / "iv2_embd" / f"{clip.uuid}.pickle"
    with embedding_pickle_path.open("rb") as infile:
        stored_embedding = pickle.load(infile)  # noqa: S301 - reading data produced within the test
    npt.assert_array_equal(stored_embedding, np.array([0.1, 0.2], dtype=np.float32))

    embedding_parquet_path = output_dir / "iv2_embd_parquet" / f"{video_uuid}_0.parquet"
    embedding_df = pd.read_parquet(embedding_parquet_path)
    assert len(embedding_df) == 1
    assert embedding_df.iloc[0]["id"] == str(clip.uuid)
    npt.assert_allclose(np.array(embedding_df.iloc[0]["embedding"]), np.array([0.1, 0.2], dtype=np.float32))


@pytest.mark.parametrize(
    ("embedding_algorithm", "expected_stem"),
    [
        ("internvideo2", "iv2_embd"),
        ("cosmos-embed1-224p", "ce1_embd_224p"),
        ("cosmos-embed1-336p", "ce1_embd_336p"),
        ("cosmos-embed1-448p", "ce1_embd_448p"),
        ("openai", "openai_embd"),
    ],
)
def test_embedding_output_paths_per_algorithm(embedding_algorithm: str, expected_stem: str) -> None:
    """Each algorithm (including per-resolution cosmos-embed1 variants) gets its own output directory."""
    root = "/data/out"
    assert ClipWriterStage.get_output_path_embds(root, embedding_algorithm) == f"{root}/{expected_stem}"
    assert ClipWriterStage.get_output_path_embd_parquets(root, embedding_algorithm) == f"{root}/{expected_stem}_parquet"
    assert ClipWriterStage.get_output_path_embd_lance(root, embedding_algorithm) == f"{root}/{expected_stem}_lance"
    assert (
        ClipWriterStage.get_output_path_embd_lance_fragments(root, embedding_algorithm)
        == f"{root}/{expected_stem}_lance_fragments"
    )
    assert (
        ClipWriterStage.get_output_path_embd_lance_fragments_processed(root, embedding_algorithm)
        == f"{root}/{expected_stem}_lance_fragments_processed"
    )


def test_cosmos_embed1_variants_write_to_distinct_paths() -> None:
    """Switching cosmos-embed1 resolution against the same output path must not silently overwrite."""
    root = "/data/out"
    paths = {
        variant: ClipWriterStage.get_output_path_embds(root, f"cosmos-embed1-{variant}")
        for variant in ("224p", "336p", "448p")
    }
    assert len(set(paths.values())) == len(paths)


def test_process_data_writes_expected_local_outputs(tmp_path: Path) -> None:
    """End-to-end validation for per-clip assets and metadata aggregation."""
    stage, task, clip, main_window, output_dir = _stage_with_main_clip(tmp_path)
    video = task.video

    result = stage.process_data([task])
    assert result == [task]

    _assert_payloads_cleared(clip, main_window)

    clip_mp4_path = output_dir / "clips" / f"{clip.uuid}.mp4"
    assert clip_mp4_path.read_bytes() == b"clip-bytes"

    preview_path = output_dir / "previews" / str(clip.uuid) / "0_30.webp"
    assert preview_path.read_bytes() == b"webp-content"

    clip_meta_path = output_dir / "metas" / "v0" / f"{clip.uuid}.json"
    clip_metadata = _read_json(clip_meta_path)
    assert clip_metadata["span_uuid"] == str(clip.uuid)
    assert clip_metadata["clip_location"].endswith(f"clips/{clip.uuid}.mp4")
    assert clip_metadata["filtered_windows"] == [
        {"start_frame": 0, "end_frame": 30, "qwen_rejection_reasons": "too blurry"}
    ]
    assert clip_metadata["windows"] == [
        {
            "start_frame": 0,
            "end_frame": 30,
            "qwen_caption": "main caption",
            "qwen_plus_enhanced_caption": "enhanced view",
        }
    ]
    assert clip_metadata["valid"] is True
    assert clip_metadata["has_caption"] is True

    video_uuid = ClipWriterStage.get_video_uuid(video.input_path)
    video_meta = _read_json(output_dir / "processed_videos" / "video.mp4.json")
    assert video_meta["video"] == video.input_path
    assert video_meta["num_total_clips"] == 1

    clip_chunk_meta = _read_json(output_dir / "processed_clip_chunks" / "video.mp4_0.json")
    assert clip_chunk_meta["num_clips_transcoded"] == 1
    assert clip_chunk_meta["num_clips_with_embeddings"] == 1
    assert clip_chunk_meta["num_clips_with_caption"] == 1
    assert clip_chunk_meta["num_clips_with_webp"] == 1
    assert clip_chunk_meta["max_clip_duration"] == pytest.approx(2.0)
    assert clip_chunk_meta["all_windows"][str(clip.uuid)] == {"0_30": "main caption"}
    assert clip_chunk_meta["all_windows_enhanced_caption"][str(clip.uuid)] == {"0_30": "enhanced view"}

    _assert_embeddings_written(output_dir, clip, video_uuid)

    cds_parquet_path = output_dir / "cds_parquet" / f"{video_uuid}_0.parquet"
    cds_df = pd.read_parquet(cds_parquet_path)
    assert len(cds_df) == 1
    assert cds_df.iloc[0]["id"] == str(clip.uuid)
    npt.assert_allclose(np.array(cds_df.iloc[0]["embedding"]), np.array([0.1, 0.2], dtype=np.float32))
    cds_meta = json.loads(cds_df.iloc[0]["$meta"])
    assert cds_meta["model_name"] == "internvideo2"
    assert cds_meta["model_version"] == "v1"
    assert cds_meta["caption"] == "main caption"
    assert cds_meta["clip_location"].endswith(f"clips/{clip.uuid}.mp4")


def test_process_data_writes_filter_window_errors(tmp_path: Path) -> None:
    """Filter-window errors should be persisted in clip metadata."""
    stage, task, clip, main_window, output_dir = _stage_with_main_clip(tmp_path)
    clip.filter_windows[0].errors["qwen"] = "malformed_model_output"

    result = stage.process_data([task])
    assert result == [task]

    _assert_payloads_cleared(clip, main_window)

    clip_meta_path = output_dir / "metas" / "v0" / f"{clip.uuid}.json"
    clip_metadata = _read_json(clip_meta_path)
    assert clip_metadata["filtered_windows"] == [
        {
            "start_frame": 0,
            "end_frame": 30,
            "qwen_rejection_reasons": "too blurry",
            "errors": {"qwen": "malformed_model_output"},
        }
    ]


def test_single_cam_clip_path_uses_flat_structure(tmp_path: Path) -> None:
    """Single-cam must write clips to clips/{uuid}.mp4 (relative_path empty)."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input-video")
    stage = _create_stage(output_dir, input_dir)
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=bytes_to_numpy(b"clip-bytes"),
        windows=[Window(start_frame=0, end_frame=30, caption={"qwen": "cap"}, caption_status="success")],
    )
    video = _build_video(video_path, clip, relative_path="")
    task = SplitPipeTask(session_id="test-session", video=video)
    stage.process_data([task])
    flat_path = output_dir / "clips" / f"{clip.uuid}.mp4"
    assert flat_path.exists(), "single-cam must use flat path clips/{uuid}.mp4"
    assert flat_path.read_bytes() == b"clip-bytes"


def test_multicam_style_clip_path_uses_subdir(tmp_path: Path) -> None:
    """When relative_path is set, clip MP4 is written to clips/{uuid}/{relative_path}.mp4."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input-video")
    stage = _create_stage(output_dir, input_dir)
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=bytes_to_numpy(b"clip-bytes"),
        windows=[Window(start_frame=0, end_frame=30, caption={"qwen": "cap"}, caption_status="success")],
    )
    video = _build_video(video_path, clip, relative_path="video")
    task = SplitPipeTask(session_id="test-session", video=video)
    stage.process_data([task])
    subdir_path = output_dir / "clips" / str(clip.uuid) / "video.mp4"
    assert subdir_path.exists(), "multi-cam style must use clips/{uuid}/{relative_path}.mp4"
    assert subdir_path.read_bytes() == b"clip-bytes"


def test_multicam_primary_only_metadata_no_overwrite(tmp_path: Path) -> None:
    """Multi-cam: per-clip metadata and embeddings are primary-only; secondary must not overwrite."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    primary_path = input_dir / "session" / "cam1.mp4"
    secondary_path = input_dir / "session" / "cam2.mp4"
    primary_path.parent.mkdir(parents=True)
    primary_path.write_bytes(b"primary-video")
    secondary_path.write_bytes(b"secondary-video")

    stage = _create_stage(output_dir, input_dir)
    shared_uuid = uuid.uuid4()

    primary_clip = Clip(
        uuid=shared_uuid,
        source_video=primary_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=bytes_to_numpy(b"primary-clip-bytes"),
        windows=[Window(start_frame=0, end_frame=30, caption={"qwen": "primary caption"}, caption_status="success")],
    )
    primary_clip.intern_video_2_embedding = np.array([1.0, 2.0], dtype=np.float32)

    secondary_clip = Clip(
        uuid=shared_uuid,
        source_video=secondary_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=bytes_to_numpy(b"secondary-clip-bytes"),
        windows=[Window(start_frame=0, end_frame=30, caption={"qwen": "secondary caption"}, caption_status="success")],
    )
    secondary_clip.intern_video_2_embedding = np.array([9.0, 9.0], dtype=np.float32)

    primary_video = _build_video(primary_path, primary_clip, relative_path="session/cam1")
    secondary_video = _build_video(secondary_path, secondary_clip, relative_path="session/cam2")
    task = SplitPipeTask(session_id="test-session", videos=[primary_video, secondary_video])

    stage.process_data([task])

    # Both MP4s written (per-camera paths)
    assert (output_dir / "clips" / str(shared_uuid) / "session" / "cam1.mp4").read_bytes() == b"primary-clip-bytes"
    assert (output_dir / "clips" / str(shared_uuid) / "session" / "cam2.mp4").read_bytes() == b"secondary-clip-bytes"

    # Per-clip metadata: primary only; must not be overwritten by secondary
    meta_path = output_dir / "metas" / "v0" / f"{shared_uuid}.json"
    meta = _read_json(meta_path)
    assert meta["windows"][0]["qwen_caption"] == "primary caption", "metadata must come from primary, not secondary"
    assert meta["source_video"] == primary_path.as_posix()

    # Embedding: primary only
    emb_path = output_dir / "iv2_embd" / f"{shared_uuid}.pickle"
    with emb_path.open("rb") as f:
        emb = pickle.load(f)  # noqa: S301
    npt.assert_allclose(emb, np.array([1.0, 2.0], dtype=np.float32))


def test_chunked_metadata_writes_group_jsonl(tmp_path: Path) -> None:
    """Ensure chunked metadata buffering emits JSONL records."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clip_info_in_chunks=True,
        upload_cds_parquet=False,
        generate_embeddings=False,
    )

    window = Window(
        start_frame=0,
        end_frame=15,
        caption={"qwen": "chunk caption"},
        caption_status="success",
    )
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 1.5),
        encoded_data=bytes_to_numpy(b"data"),
        windows=[window],
    )

    video = _build_video(video_path, clip)
    task = SplitPipeTask(session_id="test-session", video=video)

    stage.process_data([task])
    consolidate_lance_fragments(str(output_dir), "default")

    per_clip_meta_path = output_dir / "metas" / "v0" / f"{clip.uuid}.json"
    assert not per_clip_meta_path.exists()

    video_uuid = ClipWriterStage.get_video_uuid(video.input_path)
    jsonl_path = output_dir / "metas_jsonl" / "v0" / f"{video_uuid}_0.jsonl"
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 1

    chunk_record = json.loads(lines[0])
    assert chunk_record["span_uuid"] == str(clip.uuid)
    assert chunk_record["has_caption"] is True
    assert chunk_record["windows"] == [{"start_frame": 0, "end_frame": 15, "qwen_caption": "chunk caption"}]
    assert chunk_record["clip_location"].endswith(f"clips/{clip.uuid}.mp4")

    chunk_stats = _read_json(output_dir / "processed_clip_chunks" / "video.mp4_0.json")
    assert chunk_stats["num_clips_with_embeddings"] == 0
    assert chunk_stats["num_clips_with_caption"] == 1


def test_chunked_metadata_writes_lance_dataset(tmp_path: Path) -> None:
    """Verify chunked metadata optionally writes to Lance alongside JSONL."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clip_info_in_lance=True,
        upload_cds_parquet=False,
        generate_embeddings=False,
    )

    window = Window(
        start_frame=5,
        end_frame=25,
        caption={"qwen": "lance caption"},
        caption_status="success",
    )
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.5),
        encoded_data=bytes_to_numpy(b"data"),
        windows=[window],
    )

    video = _build_video(video_path, clip)
    task = SplitPipeTask(session_id="test-session", video=video)

    stage.process_data([task])
    consolidate_lance_fragments(str(output_dir), "default")

    video_uuid = ClipWriterStage.get_video_uuid(video.input_path)
    lance_root = output_dir / "lance" / "v0"
    dataset = lance.dataset(lance_root.as_posix())
    rows = dataset.to_table().to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["span_uuid"] == str(clip.uuid)
    assert row["video_uuid"] == str(video_uuid)
    assert row["clip_chunk_index"] == 0
    assert row["windows"][0]["qwen_caption"] == "lance caption"


def test_lance_consolidation_is_idempotent(tmp_path: Path) -> None:
    """Verify calling consolidate_lance_fragments multiple times appends correctly and archives sidecars."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clip_info_in_lance=True,
        upload_cds_parquet=False,
        generate_embeddings=False,
    )

    # First clip and consolidation
    window1 = Window(start_frame=0, end_frame=10, caption={"qwen": "first caption"}, caption_status="success")
    clip1 = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 1.0),
        encoded_data=bytes_to_numpy(b"data1"),
        windows=[window1],
    )
    video1 = _build_video(video_path, clip1, clip_chunk_index=0)
    task1 = SplitPipeTask(session_id="test-session", video=video1)
    stage.process_data([task1])
    consolidate_lance_fragments(str(output_dir), "default")

    lance_root = output_dir / "lance" / "v0"
    dataset = lance.dataset(lance_root.as_posix())
    assert len(dataset.to_table()) == 1

    # Verify first sidecar was archived
    staging_dir = output_dir / "lance_fragments" / "v0"
    processed_dir = output_dir / "processed_lance_fragments" / "v0"
    assert len(list(staging_dir.glob("*.json"))) == 0
    assert len(list(processed_dir.glob("*.json"))) == 1

    # Second clip (different chunk) and consolidation
    window2 = Window(start_frame=10, end_frame=20, caption={"qwen": "second caption"}, caption_status="success")
    clip2 = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(1.0, 2.0),
        encoded_data=bytes_to_numpy(b"data2"),
        windows=[window2],
    )
    video2 = _build_video(video_path, clip2, clip_chunk_index=1)
    task2 = SplitPipeTask(session_id="test-session", video=video2)
    stage.process_data([task2])
    consolidate_lance_fragments(str(output_dir), "default")

    # Verify dataset was appended (2 rows total)
    dataset = lance.dataset(lance_root.as_posix())
    rows = dataset.to_table().to_pylist()
    assert len(rows) == 2
    span_uuids = {row["span_uuid"] for row in rows}
    assert str(clip1.uuid) in span_uuids
    assert str(clip2.uuid) in span_uuids

    # Verify second sidecar was also archived
    assert len(list(staging_dir.glob("*.json"))) == 0
    assert len(list(processed_dir.glob("*.json"))) == 2

    # Third consolidation with no new sidecars should be a no-op
    consolidate_lance_fragments(str(output_dir), "default")
    dataset = lance.dataset(lance_root.as_posix())
    assert len(dataset.to_table()) == 2


def test_archive_processed_sidecars_raises_on_remote_delete_failure() -> None:
    """Ensure cleanup errors for remote sidecars surface instead of being swallowed."""
    staging_root = "s3://bucket/staging"
    processed_root = "s3://bucket/processed"
    sidecar_name = "chunk.json"
    payload = {"fragments": [], "schema_b64": base64.b64encode(b"schema").decode("ascii")}
    staged_path = storage_utils.get_full_path(staging_root, sidecar_name)
    staging_client = _FailingDeleteClient({str(staged_path): json.dumps(payload).encode("utf-8")})
    processed_client = _FailingDeleteClient()

    with pytest.raises(RuntimeError, match="Failed to delete remote sidecar"):
        _archive_processed_sidecars(
            [sidecar_name],
            staging_root,
            processed_root,
            staging_client,
            processed_client,
        )

    processed_path = storage_utils.get_full_path(processed_root, sidecar_name)
    assert str(processed_path) in processed_client.objects


def test_per_window_dataset_assets_written(tmp_path: Path) -> None:
    """Verify per-window dataset assets are written when enabled."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clips=False,
        upload_cds_parquet=False,
        generate_embeddings=False,
        generate_cosmos_predict_dataset="enable",
    )

    window = Window(
        start_frame=0,
        end_frame=20,
        mp4_bytes=b"window-mp4",
        caption={"qwen": "dataset caption"},
        caption_status="success",
        t5_xxl_embedding={"default": np.array([1, 2], dtype=np.int32)},
    )
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=bytes_to_numpy(b"clip-bytes"),
        windows=[window],
    )

    video = _build_video(video_path, clip)
    task = SplitPipeTask(session_id="test-session", video=video)

    stage.process_data([task])

    dataset_root = output_dir / "cosmos_predict2_video2world_dataset"
    video_file = dataset_root / "videos" / f"{clip.uuid}_0_20.mp4"
    meta_file = dataset_root / "metas" / f"{clip.uuid}_0_20.txt"
    t5_file = dataset_root / "t5_xxl" / f"{clip.uuid}_0_20.pickle"

    assert video_file.read_bytes() == b"window-mp4"
    assert meta_file.read_text() == "dataset caption"
    with t5_file.open("rb") as infile:
        stored_t5 = pickle.load(infile)  # noqa: S301 - reading data produced within the test
    assert len(stored_t5) == 1
    npt.assert_array_equal(stored_t5[0], np.array([1, 2], dtype=np.int32))

    assert window.mp4_bytes.resolve() is None


def test_filtered_clips_cleaned_up_after_processing(tmp_path: Path) -> None:
    """Verify that filtered_clips have intermediate data released after processing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input-video")

    stage = _create_stage(output_dir, input_dir, generate_embeddings=False, upload_cds_parquet=False)

    kept_clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(0.0, 2.0),
        encoded_data=bytes_to_numpy(b"kept-bytes"),
        windows=[Window(start_frame=0, end_frame=30, caption={"qwen": "cap"}, caption_status="success")],
    )
    filtered_window = Window(
        start_frame=0,
        end_frame=30,
        mp4_bytes=b"filtered-window-mp4",
        caption={"qwen": "filtered caption"},
        caption_status="success",
        enhanced_caption={"qwen_plus": "enhanced filtered"},
        webp_bytes=b"filtered-webp",
    )
    filtered_clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(2.0, 4.0),
        encoded_data=bytes_to_numpy(b"filtered-bytes"),
        windows=[filtered_window],
    )
    filtered_clip.intern_video_2_embedding = np.array([0.5], dtype=np.float32)
    filtered_clip.cosmos_embed1_embedding = np.array([0.6], dtype=np.float32)
    filtered_clip.openai_embedding = np.array([0.7], dtype=np.float32)

    metadata = VideoMetadata(
        height=1080,
        width=1920,
        framerate=30.0,
        num_frames=120,
        duration=4.0,
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac",
    )
    video = Video(
        input_video=video_path,
        metadata=metadata,
        clips=[kept_clip],
        filtered_clips=[filtered_clip],
        num_total_clips=2,
        num_clip_chunks=1,
    )
    task = SplitPipeTask(session_id="test-session", video=video)

    stage.process_data([task])

    assert filtered_clip.encoded_data.resolve() is None
    assert filtered_clip.intern_video_2_embedding is None
    assert filtered_clip.cosmos_embed1_embedding is None
    assert filtered_clip.openai_embedding is None
    assert filtered_window.mp4_bytes.resolve() is None
    assert filtered_window.caption == {}
    assert filtered_window.enhanced_caption == {}
    assert filtered_window.webp_bytes.resolve() is None

    # Filtered clip MP4 should still be written to disk
    filtered_mp4 = output_dir / "filtered_clips" / f"{filtered_clip.uuid}.mp4"
    assert filtered_mp4.read_bytes() == b"filtered-bytes"


def test_video_errors_written_to_error_path(tmp_path: Path) -> None:
    """Verify that videos with errors write to video_errors path and skip normal chunk metadata."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    errors = {"captioning": "test error"}
    input_dir.mkdir()
    video_path = input_dir / "video.mp4"
    video_path.write_bytes(b"input-video")

    stage = _create_stage(
        output_dir,
        input_dir,
        upload_clips=False,
        upload_cds_parquet=False,
        generate_embeddings=False,
    )

    # Create a video with errors (no clips)
    metadata = VideoMetadata(
        height=1080,
        width=1920,
        framerate=30.0,
        num_frames=60,
        duration=2.0,
        video_codec="h264",
        pixel_format="yuv420p",
        audio_codec="aac",
    )
    video = Video(
        input_video=video_path,
        metadata=metadata,
        clips=[],
        filtered_clips=[],
        num_total_clips=0,
        num_clip_chunks=1,
        clip_chunk_index=0,
        errors=errors,
    )
    task = SplitPipeTask(session_id="test-session", video=video)

    stage.process_data([task])

    # Verify that normal chunk metadata is NOT written
    chunk_meta_path = output_dir / "processed_clip_chunks" / "video.mp4_0.json"
    assert not chunk_meta_path.exists()

    # Verify that error file IS written
    error_path = output_dir / "video_errors" / "video.mp4_0.json"
    assert error_path.exists()

    error_data = _read_json(error_path)
    assert error_data["video"] == video_path.as_posix()
    assert error_data["clip_chunk_index"] == 0
    assert error_data["errors"] == errors

    # Verify that video-level metadata is also NOT written (since errors exist)
    video_meta_path = output_dir / "processed_videos" / "video.mp4.json"
    assert not video_meta_path.exists()


# ---------------------------------------------------------------------------
# caption_status — Layer 1 (has_caption) and Layer 2 (num_with_caption) tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("caption_status", "caption_value", "expected_has_caption"),
    [
        ("success", "A great video.", True),
        ("truncated", "A partial caption.", True),
        ("blocked", "unexpected text", False),
        ("error", "unexpected text", False),
        ("error", None, False),
    ],
    ids=["success", "truncated", "blocked", "error_with_text", "error_no_text"],
)
def test_make_clip_metadata_has_caption(
    tmp_path: Path,
    caption_status: str | None,
    caption_value: str | None,
    *,
    expected_has_caption: bool,
) -> None:
    """has_caption depends on caption_status only."""
    stage = _create_stage(tmp_path / "out", tmp_path / "in")

    caption = {"qwen": caption_value} if caption_value is not None else {}
    window = Window(start_frame=0, end_frame=10, caption=caption, caption_status=caption_status)
    clip = Clip(uuid=uuid.uuid4(), source_video="v.mp4", span=(0.0, 1.0), windows=[window])
    video_meta = VideoMetadata(height=1080, width=1920, framerate=30.0, num_frames=30, duration=1.0, video_codec="h264")

    data = stage._make_clip_metadata(clip, video_meta)
    assert data["has_caption"] is expected_has_caption


@pytest.mark.parametrize(
    ("caption_status", "caption_value", "expected_count"),
    [
        ("success", "A great video.", 1),
        ("truncated", "A partial caption.", 1),
        ("blocked", "unexpected text", 0),
        ("error", "unexpected text", 0),
        ("error", None, 0),
    ],
    ids=["success", "truncated", "blocked", "error_with_text", "error_no_text"],
)
def test_write_clip_metadata_num_with_caption(
    tmp_path: Path,
    caption_status: str | None,
    caption_value: str | None,
    expected_count: int,
) -> None:
    """num_with_caption metric matches has_caption, not raw caption-key presence."""
    stage = _create_stage(tmp_path / "out", tmp_path / "in")

    caption = {"qwen": caption_value} if caption_value is not None else {}
    window = Window(start_frame=0, end_frame=10, caption=caption, caption_status=caption_status)
    clip = Clip(uuid=uuid.uuid4(), source_video="v.mp4", span=(0.0, 1.0), windows=[window])
    video_meta = VideoMetadata(height=1080, width=1920, framerate=30.0, num_frames=30, duration=1.0, video_codec="h264")

    clip_stats = stage._write_clip_metadata(clip, video_meta)
    assert clip_stats.num_with_caption == expected_count


# ---------------------------------------------------------------------------
# caption_status — Layer 3 (per-window export gate) tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("caption_status", "caption_value", "expect_export"),
    [
        ("success", "A great video.", True),
        ("truncated", "A partial caption.", True),
        ("blocked", "unexpected text", False),
        ("error", "unexpected text", False),
        ("error", None, False),
    ],
    ids=["success", "truncated", "blocked", "error_with_text", "error_no_text"],
)
def test_write_per_window_data_export_gate(
    tmp_path: Path,
    caption_status: str | None,
    caption_value: str | None,
    *,
    expect_export: bool,
) -> None:
    """Windows without a usable caption are skipped; windows with one are exported."""
    output_dir = tmp_path / "out"
    stage = _create_stage(
        output_dir,
        tmp_path / "in",
        generate_cosmos_predict_dataset="enable",
    )

    caption = {"qwen": caption_value} if caption_value is not None else {}
    t5_embed = {"default": np.array([1, 2, 3], dtype=np.int32)}
    window = Window(
        start_frame=0,
        end_frame=10,
        mp4_bytes=b"mp4data",
        caption=caption,
        caption_status=caption_status,
        t5_xxl_embedding=t5_embed,
    )
    clip = Clip(uuid=uuid.uuid4(), source_video="v.mp4", span=(0.0, 1.0), windows=[window])

    stage._write_per_window_data(clip)

    mp4_out = output_dir / "cosmos_predict2_video2world_dataset" / "videos" / f"{clip.uuid}_0_10.mp4"
    assert mp4_out.exists() is expect_export
