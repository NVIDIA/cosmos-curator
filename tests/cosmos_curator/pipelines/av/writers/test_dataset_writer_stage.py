# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Tests for dataset_writer_stage module."""

import copy
import io
import json
import pickle
import tarfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cosmos_curator.core.utils.db.database_types import EnvType, PostgresDB
from cosmos_curator.core.utils.infra.performance_utils import StagePerfStats
from cosmos_curator.pipelines.av.utils.av_data_info import CAMERA_MAPPING
from cosmos_curator.pipelines.av.utils.av_data_model import AvSample, AvShardingTask
from cosmos_curator.pipelines.av.utils.av_pipe_input import WINDOWS_PER_CLIP
from cosmos_curator.pipelines.av.writers import dataset_writer_stage


def _read_tar_content(buffer: bytes) -> dict[str, bytes]:
    """Return the tar entries (name -> payload) for the given buffer."""
    result: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(buffer), mode="r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            assert extracted is not None
            result[member.name] = extracted.read()
    return result


@pytest.fixture
def dummy_postgres_db() -> PostgresDB:
    """Create simple Postgres configuration for stages under test."""
    return PostgresDB(
        EnvType.LOCAL,
        "endpoint",
        "database",
        "user",
        "password",
    )


@pytest.fixture
def test_camera_format(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a compact camera mapping for tests."""
    camera_format_id = "TEST_CAM"
    camera_mapping = {
        "camera_id_extractor": {
            "delimiter": "-",
            "index": 1,
        },
        "camera_id_mapping_cosmos": {
            1: 0,
            2: 1,
        },
        "camera_name_mapping_cosmos": {
            1: "camera_front",
            2: "camera_rear",
        },
        "all_timestamp_files": [],
        "camera_id_for_vri_caption": [1],
    }
    monkeypatch.setitem(CAMERA_MAPPING, camera_format_id, camera_mapping)
    return camera_format_id


@pytest.fixture
def fake_s3_clients(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Provide a hook to capture S3 client creations."""
    clients: list[object] = []

    def _fake_create_s3_client(*, target_path: str) -> object:  # noqa: ARG001
        client = object()
        clients.append(client)
        return client

    monkeypatch.setattr(dataset_writer_stage.s3_client, "create_s3_client", _fake_create_s3_client)
    return clients


def test_create_tar_bytes() -> None:
    """_create_tar_bytes should bundle the provided samples."""
    tar_bytes = dataset_writer_stage._create_tar_bytes(
        [
            (b"alpha", "file_a.bin"),
            (b"beta", "dir/file_b.txt"),
        ]
    )
    content = _read_tar_content(tar_bytes)

    assert content == {
        "file_a.bin": b"alpha",
        "dir/file_b.txt": b"beta",
    }


def test_clip_packaging_stage_writes_expected_tar(  # noqa: PLR0915
    tmp_path: Path,
    dummy_postgres_db: PostgresDB,
    test_camera_format: str,
    fake_s3_clients: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ClipPackagingStage should package clips, timestamps, and trajectories into a tar archive."""
    camera_mapping = CAMERA_MAPPING[test_camera_format]["camera_name_mapping_cosmos"]
    camera_ids = list(camera_mapping.keys())

    clip_session_uuid = uuid.uuid4()
    clip_uuids = [uuid.uuid4() for _ in camera_ids]
    clip_urls = [f"memory://clip-{idx}" for idx, _ in enumerate(camera_ids)]
    clip_payloads = {clip_urls[idx]: f"clip-{idx}".encode() for idx in range(len(camera_ids))}
    clip_timestampss_ms = [
        np.array([1000, 2000], dtype=np.int64).tobytes(),
        np.array([3000, 4000], dtype=np.int64).tobytes(),
    ]
    trajectory_urls = [f"memory://traj-{idx}" for idx in range(len(camera_ids))]
    trajectory_payloads = {trajectory_urls[idx]: f"traj-{idx}".encode() for idx in range(len(camera_ids))}
    window_captions = [
        ["caption-front-v0", "caption-front-v1"],
        ["caption-rear-v0", "caption-rear-v1"],
    ]
    window_start_frames = [
        [0, 50],
        [10, 60],
    ]
    window_end_frames = [
        [5, 55],
        [15, 65],
    ]
    t5_urls = [f"memory://t5-{idx}" for idx in range(len(camera_ids))]

    sample = AvSample(
        clip_session_uuid=clip_session_uuid,
        camera_ids=camera_ids,
        clip_uuids=clip_uuids,
        clip_urls=clip_urls,
        clip_timestampss_ms=clip_timestampss_ms,
        window_captions=window_captions,
        window_start_frames=window_start_frames,
        window_end_frames=window_end_frames,
        t5_urls=t5_urls,
        trajectory_urls=trajectory_urls,
    )
    task = AvShardingTask(part_num=7, samples=[sample])

    payloads: dict[str, bytes] = {}
    payloads.update(clip_payloads)
    payloads.update(trajectory_payloads)

    def _fake_read_bytes(path: str, client: object) -> bytes:  # noqa: ARG001
        return payloads[path]

    write_calls: list[dict[str, Any]] = []

    def _fake_write_bytes(  # noqa: PLR0913
        buffer: bytes,
        dest: Path,
        desc: str,
        source_video: str,
        *,
        verbose: bool,
        client: object,
    ) -> None:
        write_calls.append(
            {
                "buffer": buffer,
                "dest": dest,
                "desc": desc,
                "source": source_video,
                "verbose": verbose,
                "client": client,
            }
        )

    monkeypatch.setattr(dataset_writer_stage, "read_bytes", _fake_read_bytes)
    monkeypatch.setattr(dataset_writer_stage, "write_bytes", _fake_write_bytes)

    stage = dataset_writer_stage.ClipPackagingStage(
        db=dummy_postgres_db,
        camera_format_id=test_camera_format,
        dataset_name="demo_dataset",
        output_prefix=str(tmp_path),
        verbose=False,
        log_stats=True,
    )
    stage.stage_setup()
    perf_stats = StagePerfStats(process_time=0.25)
    monkeypatch.setattr(stage._timer, "log_stats", lambda _verbose=False: ("ClipPackagingStage", perf_stats))

    result = stage.process_data([task])

    assert result == [task]
    assert not task.s3_upload_error
    assert len(fake_s3_clients) == 2
    assert len(write_calls) == 1
    output_client = fake_s3_clients[1]
    call = write_calls[0]
    assert call["client"] is output_client
    expected_dest = Path(
        f"{tmp_path}/{dummy_postgres_db.env_type.value}/datasets/demo_dataset/clips/{clip_session_uuid}.tar"
    )
    assert call["dest"] == expected_dest
    assert call["desc"] == str(clip_session_uuid)
    assert call["source"] == str(clip_session_uuid)
    assert task.stage_perf["ClipPackagingStage"] is perf_stats

    content = _read_tar_content(call["buffer"])
    expected_names = []
    for idx, camera_id in enumerate(camera_ids):
        camera_name = camera_mapping[camera_id]
        expected_names.extend(
            [
                f"{clip_session_uuid}.{camera_name}.mp4",
                f"{clip_session_uuid}.{camera_name}.json",
                f"{clip_session_uuid}.{camera_name}.bin",
            ]
        )
        assert content[f"{clip_session_uuid}.{camera_name}.mp4"] == clip_payloads[clip_urls[idx]]
        timestamps = np.frombuffer(clip_timestampss_ms[idx], dtype=np.int64)
        expected_metadata = [
            {
                "frame_num": frame_idx,
                "timestamp": int(timestamp),
            }
            for frame_idx, timestamp in enumerate(timestamps)
        ]
        actual_metadata = json.loads(content[f"{clip_session_uuid}.{camera_name}.json"].decode("utf-8"))
        assert actual_metadata == expected_metadata
        assert content[f"{clip_session_uuid}.{camera_name}.bin"] == trajectory_payloads[trajectory_urls[idx]]
    assert set(content.keys()) == set(expected_names)


def test_t5_embedding_packaging_stage_e_creates_variant_tars(
    tmp_path: Path,
    dummy_postgres_db: PostgresDB,
    test_camera_format: str,
    fake_s3_clients: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5EmbeddingPackagingStageE should emit a tar per T5 variant containing embeddings and metadata."""
    camera_mapping = CAMERA_MAPPING[test_camera_format]["camera_name_mapping_cosmos"]
    camera_ids = list(camera_mapping.keys())

    clip_session_uuid = uuid.uuid4()
    clip_uuids = [uuid.uuid4() for _ in camera_ids]
    clip_urls = [f"memory://clip-{idx}" for idx in range(len(camera_ids))]
    clip_timestampss_ms = [np.array([0], dtype=np.int64).tobytes() for _ in camera_ids]
    t5_urls = [f"memory://t5-{idx}" for idx in range(len(camera_ids))]

    embeddings_per_camera = [
        [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ],
        [
            np.array([5.0, 6.0], dtype=np.float32),
            np.array([7.0, 8.0], dtype=np.float32),
        ],
    ]
    window_captions = [
        ["front-caption-v0", "front-caption-v1"],
        ["rear-caption-v0", "rear-caption-v1"],
    ]
    window_start_frames = [
        [0, 50],
        [5, 55],
    ]
    window_end_frames = [
        [10, 60],
        [15, 65],
    ]

    sample = AvSample(
        clip_session_uuid=clip_session_uuid,
        camera_ids=camera_ids,
        clip_uuids=clip_uuids,
        clip_urls=clip_urls,
        clip_timestampss_ms=clip_timestampss_ms,
        window_captions=window_captions,
        window_start_frames=window_start_frames,
        window_end_frames=window_end_frames,
        t5_urls=t5_urls,
        trajectory_urls=None,
    )
    task = AvShardingTask(part_num=3, samples=[sample])

    payloads = {t5_urls[idx]: pickle.dumps(embeddings_per_camera[idx]) for idx in range(len(camera_ids))}

    def _fake_read_bytes(path: str, client: object) -> bytes:  # noqa: ARG001
        return payloads[path]

    write_calls: list[dict[str, Any]] = []

    def _fake_write_bytes(  # noqa: PLR0913
        buffer: bytes,
        dest: Path,
        desc: str,
        source_video: str,
        *,
        verbose: bool,
        client: object,
    ) -> None:
        write_calls.append(
            {
                "buffer": buffer,
                "dest": dest,
                "desc": desc,
                "source": source_video,
                "verbose": verbose,
                "client": client,
            }
        )

    monkeypatch.setattr(dataset_writer_stage, "read_bytes", _fake_read_bytes)
    monkeypatch.setattr(dataset_writer_stage, "write_bytes", _fake_write_bytes)

    stage = dataset_writer_stage.T5EmbeddingPackagingStageE(
        db=dummy_postgres_db,
        camera_format_id=test_camera_format,
        dataset_name="demo_dataset",
        output_prefix=str(tmp_path),
        verbose=False,
        log_stats=True,
    )
    stage.stage_setup()
    perf_stats = StagePerfStats(process_time=0.1)
    monkeypatch.setattr(stage._timer, "log_stats", lambda _verbose=False: ("T5EmbeddingPackagingStageE", perf_stats))

    result = stage.process_data([task])

    assert result == [task]
    assert len(fake_s3_clients) == 2
    assert len(write_calls) == len(dataset_writer_stage.T5_VARIANTS)
    assert task.stage_perf["T5EmbeddingPackagingStageE"] is perf_stats

    output_client = fake_s3_clients[1]
    for variant_idx, variant_name in dataset_writer_stage.T5_VARIANTS.items():
        call = write_calls[variant_idx]
        assert call["client"] is output_client
        expected_dest = Path(
            f"{tmp_path}/{dummy_postgres_db.env_type.value}/datasets/demo_dataset/{variant_name}/{clip_session_uuid}.tar"
        )
        assert call["dest"] == expected_dest
        assert call["desc"] == f"{clip_session_uuid}-{variant_name}"
        assert call["source"] == f"{clip_session_uuid}-{variant_name}"

        content = _read_tar_content(call["buffer"])
        expected_names = []
        for idx, camera_id in enumerate(camera_ids):
            camera_name = camera_mapping[camera_id]
            expected_names.extend(
                [
                    f"{clip_session_uuid}.{camera_name}.bin",
                    f"{clip_session_uuid}.{camera_name}.json",
                ]
            )
            embedding = pickle.loads(content[f"{clip_session_uuid}.{camera_name}.bin"])  # noqa: S301
            assert np.array_equal(embedding, embeddings_per_camera[idx][variant_idx])
            expected_metadata = [
                str(clip_uuids[idx]),
                [window_captions[idx][variant_idx]],
                [window_start_frames[idx][variant_idx]],
                [window_end_frames[idx][variant_idx]],
            ]
            metadata = json.loads(content[f"{clip_session_uuid}.{camera_name}.json"].decode("utf-8"))
            assert metadata == expected_metadata
        assert set(content.keys()) == set(expected_names)


def test_t5_embedding_packaging_stage_h_uploads_tar_and_metadata(  # noqa: PLR0915
    tmp_path: Path,
    dummy_postgres_db: PostgresDB,
    test_camera_format: str,
    fake_s3_clients: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5EmbeddingPackagingStageH should emit tar files plus metadata and update tar mappings."""
    camera_mapping = CAMERA_MAPPING[test_camera_format]["camera_name_mapping_cosmos"]
    camera_id_mapping_cosmos = CAMERA_MAPPING[test_camera_format]["camera_id_mapping_cosmos"]
    camera_ids = list(camera_mapping.keys())

    clip_session_uuid = uuid.uuid4()
    clip_uuids = [uuid.uuid4() for _ in camera_ids]
    clip_urls = [f"memory://clip-{idx}" for idx in range(len(camera_ids))]
    clip_timestampss_ms = [np.array([0], dtype=np.int64).tobytes() for _ in camera_ids]
    t5_urls = [f"memory://t5-{idx}" for idx in range(len(camera_ids))]

    embeddings_per_camera = [
        [
            np.array([9.0, 10.0], dtype=np.float32),
            np.array([11.0, 12.0], dtype=np.float32),
        ],
        [
            np.array([13.0, 14.0], dtype=np.float32),
            np.array([15.0, 16.0], dtype=np.float32),
        ],
    ]
    window_captions = [
        ["front-caption-v0", "front-caption-v1"],
        ["rear-caption-v0", "rear-caption-v1"],
    ]
    window_start_frames = [
        [0, 100],
        [10, 110],
    ]
    window_end_frames = [
        [5, 105],
        [15, 115],
    ]

    sample = AvSample(
        clip_session_uuid=clip_session_uuid,
        camera_ids=camera_ids,
        clip_uuids=clip_uuids,
        clip_urls=clip_urls,
        clip_timestampss_ms=clip_timestampss_ms,
        window_captions=window_captions,
        window_start_frames=window_start_frames,
        window_end_frames=window_end_frames,
        t5_urls=t5_urls,
        trajectory_urls=None,
    )
    task = AvShardingTask(part_num=5, samples=[sample])

    payloads = {t5_urls[idx]: pickle.dumps(embeddings_per_camera[idx]) for idx in range(len(camera_ids))}

    def _fake_read_bytes(path: str, client: object) -> bytes:  # noqa: ARG001
        return payloads[path]

    write_bytes_calls: list[dict[str, Any]] = []
    write_json_calls: list[dict[str, Any]] = []

    def _fake_write_bytes(  # noqa: PLR0913
        buffer: bytes,
        dest: Path,
        desc: str,
        source_video: str,
        *,
        verbose: bool,
        client: object,
    ) -> None:
        write_bytes_calls.append(
            {
                "buffer": buffer,
                "dest": dest,
                "desc": desc,
                "source": source_video,
                "verbose": verbose,
                "client": client,
            }
        )

    def _fake_write_json(  # noqa: PLR0913
        data: dict[str, Any],
        dest: Path,
        desc: str,
        source_video: str,
        *,
        verbose: bool,
        client: object,
    ) -> None:
        write_json_calls.append(
            {
                "data": copy.deepcopy(data),
                "dest": dest,
                "desc": desc,
                "source": source_video,
                "verbose": verbose,
                "client": client,
            }
        )

    monkeypatch.setattr(dataset_writer_stage, "read_bytes", _fake_read_bytes)
    monkeypatch.setattr(dataset_writer_stage, "write_bytes", _fake_write_bytes)
    monkeypatch.setattr(dataset_writer_stage, "write_json", _fake_write_json)

    stage = dataset_writer_stage.T5EmbeddingPackagingStageH(
        db=dummy_postgres_db,
        camera_format_id=test_camera_format,
        dataset_name="demo_dataset",
        output_prefix=str(tmp_path),
        verbose=False,
        log_stats=True,
    )
    stage.stage_setup()
    perf_stats = StagePerfStats(process_time=0.2)
    monkeypatch.setattr(stage._timer, "log_stats", lambda _verbose=False: ("T5EmbeddingPackagingStageH", perf_stats))

    result = stage.process_data([task])

    assert result == [task]
    assert len(fake_s3_clients) == 2
    assert len(write_bytes_calls) == WINDOWS_PER_CLIP
    assert len(write_json_calls) == WINDOWS_PER_CLIP
    assert task.stage_perf["T5EmbeddingPackagingStageH"] is perf_stats

    output_client = fake_s3_clients[1]
    for variant_idx in range(WINDOWS_PER_CLIP):
        variant_name = dataset_writer_stage.T5_VARIANTS[variant_idx]
        bytes_call = write_bytes_calls[variant_idx]
        json_call = write_json_calls[variant_idx]

        assert bytes_call["client"] is output_client
        assert json_call["client"] is output_client

        expected_tar_dest = stage._get_tar_url(variant_name, task.part_num, 0)
        expected_json_dest = stage._get_metadata_url(variant_name, task.part_num, 0)

        assert bytes_call["dest"] == expected_tar_dest
        assert json_call["dest"] == expected_json_dest
        assert bytes_call["desc"] == f"t5_{0:06d}"
        assert json_call["desc"] == f"t5_{0:06d}"
        assert bytes_call["source"] == f"part_{task.part_num:06d}"
        assert json_call["source"] == f"part_{task.part_num:06d}"

        content = _read_tar_content(bytes_call["buffer"])
        expected_names = []
        for idx, camera_id in enumerate(camera_ids):
            camera_name = camera_mapping[camera_id]
            expected_names.append(f"{clip_session_uuid}.{camera_name}.bin")
            embedding = pickle.loads(content[f"{clip_session_uuid}.{camera_name}.bin"])  # noqa: S301
            assert np.array_equal(embedding, embeddings_per_camera[idx][variant_idx])
        assert set(content.keys()) == set(expected_names)

        expected_metadata = {
            str(clip_session_uuid): {
                camera_id_mapping_cosmos[camera_ids[idx]]: [
                    "demo_dataset",
                    [window_captions[idx][variant_idx]],
                    [window_start_frames[idx][variant_idx]],
                    [window_end_frames[idx][variant_idx]],
                ]
                for idx in range(len(camera_ids))
            }
        }
        assert json_call["data"] == expected_metadata

        expected_mapping_path = str(expected_tar_dest)
        for _idx, camera_id in enumerate(camera_ids):
            camera_name = camera_mapping[camera_id]
            mapping_key = f"{clip_session_uuid}.{camera_name}"
            assert task.tar_mappings[variant_idx][mapping_key] == expected_mapping_path
