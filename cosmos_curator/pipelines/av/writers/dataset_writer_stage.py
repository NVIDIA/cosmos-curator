# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Module for packaging and writing dataset components.

This module provides stages for packaging and writing various dataset components
including video clips, T5 embeddings, and associated metadata to storage (local or S3).
"""

import io
import json
import pickle
import tarfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage
from cosmos_curator.core.utils.db.database_types import PostgresDB
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.storage import s3_client
from cosmos_curator.core.utils.storage.s3_client import is_s3path
from cosmos_curator.core.utils.storage.storage_utils import read_bytes
from cosmos_curator.core.utils.storage.writer_utils import write_bytes, write_json
from cosmos_curator.pipelines.av.utils.av_data_info import CAMERA_MAPPING
from cosmos_curator.pipelines.av.utils.av_data_model import AvShardingTask
from cosmos_curator.pipelines.av.utils.av_pipe_input import WINDOWS_PER_CLIP

# Maximum number of tar archives per part
MAX_TARS_PER_PART = 32
# Number of embeddings to store in each tar archive
_EMBEDDINGS_PER_TAR = 16

# Mapping of T5 model variants
T5_VARIANTS = {0: "t5", 1: "t5_short"}


def _create_tar_bytes(samples_to_write: list[tuple[bytes, str]]) -> bytes:
    """Create a tar archive in memory from a list of byte data and filenames.

    Args:
        samples_to_write: List of tuples containing (data_bytes, filename)

    Returns:
        Bytes of the tar archive

    """
    bytes_io = io.BytesIO()
    with tarfile.open(fileobj=bytes_io, mode="w") as tar:
        for data, filename in samples_to_write:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
    bytes_io.seek(0)
    return bytes_io.getvalue()


class ClipPackagingStage(CuratorStage):
    """Stage for packaging video clips and metadata into tar archives.

    This stage handles packaging video clips, timestamps, and trajectory data
    into tar archives for each clip session.
    """

    def __init__(  # noqa: PLR0913
        self,
        db: PostgresDB,
        camera_format_id: str,
        dataset_name: str,
        output_prefix: str,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that packages video clips and metadata into tar archives.

        Args:
            db: PostgreSQL database configuration
            camera_format_id: ID for camera format configuration
            dataset_name: Name of the dataset
            output_prefix: Base path for output files
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        """
        self._timer = StageTimer(self)
        self._camera_mapping = CAMERA_MAPPING[camera_format_id]["camera_name_mapping_cosmos"]
        self._output_prefix = output_prefix.rstrip("/")
        self._clip_prefix = f"{db.env_type.value}/datasets/{dataset_name}/clips"
        self._verbose = verbose
        self._log_stats = log_stats

    def stage_setup(self) -> None:
        """Set up S3 clients for input and output operations."""
        self._s3_client_input = s3_client.create_s3_client(target_path=self._output_prefix)
        self._s3_client_output = s3_client.create_s3_client(target_path=self._output_prefix)

    def _get_tar_url(
        self,
        clip_session_uuid: uuid.UUID,
    ) -> s3_client.S3Prefix | Path:
        """Generate a URL or path for storing tar archive data.

        Args:
            clip_session_uuid: UUID of the clip session

        Returns:
            An S3Prefix object if the output_prefix is an S3 path, otherwise a
            pathlib.Path object

        """
        full_path = f"{self._output_prefix}/{self._clip_prefix}/{clip_session_uuid}.tar"
        if is_s3path(self._output_prefix):
            return s3_client.S3Prefix(full_path)
        return Path(full_path)

    def _get_clip_name(self, clip_session_uuid: uuid.UUID, camera_name: str, file_ext: str) -> str:
        """Generate a filename for a clip component.

        Args:
            clip_session_uuid: UUID of the clip session
            camera_name: Name of the camera
            file_ext: File extension (e.g., "mp4", "json", "bin")

        Returns:
            Generated filename

        """
        return f"{clip_session_uuid}.{camera_name}.{file_ext}"

    def process_data(self, tasks: list[AvShardingTask]) -> list[AvShardingTask] | None:  # type: ignore[override]
        """Process and package clips with performance tracking.

        This method packages video clips, timestamps, and trajectory data into tar
        archives for each clip session.

        Args:
            tasks: Tasks containing clips to packag

        Returns:
            List containing the input task if successful

        Raises:
            Exception: If storage operations fail

        """
        return [self._process_task(task) for task in tasks]

    def _process_task(self, task: AvShardingTask) -> AvShardingTask:
        self._timer.reinit(self, task.get_major_size())
        # upload clips to s3
        with self._timer.time_process():
            num_uploaded_tars = 0
            try:
                for sample in task.samples:
                    clip_session_uuid = sample.clip_session_uuid
                    if any(x not in sample.camera_ids for x in self._camera_mapping):
                        logger.warning(f"clip-session {clip_session_uuid} does not have all cameras")
                        continue
                    samples_to_write = []
                    for idx in range(len(sample.camera_ids)):
                        camera_id = sample.camera_ids[idx]
                        if camera_id not in self._camera_mapping:
                            continue
                        camera_name = self._camera_mapping[camera_id]
                        # Prepare MP4 clip
                        clip_buffer = read_bytes(
                            sample.clip_urls[idx],
                            self._s3_client_input,
                        )
                        samples_to_write.append(
                            (
                                clip_buffer,
                                self._get_clip_name(clip_session_uuid, camera_name, "mp4"),
                            )
                        )
                        # Prepare timestamp JSON
                        clip_metadata = []
                        timestamps_ms = np.frombuffer(sample.clip_timestampss_ms[idx], dtype=np.int64)
                        for frame_num, timestamp_ms in enumerate(timestamps_ms):
                            clip_metadata.append(
                                {
                                    "frame_num": frame_num,
                                    "timestamp": int(timestamp_ms),
                                }
                            )
                        samples_to_write.append(
                            (
                                json.dumps(clip_metadata).encode("utf-8"),
                                self._get_clip_name(clip_session_uuid, camera_name, "json"),
                            )
                        )
                        # Prepare trajectory binary
                        if sample.trajectory_urls is not None:
                            trajectory_buffer = read_bytes(
                                sample.trajectory_urls[idx],
                                self._s3_client_input,
                            )
                            samples_to_write.append(
                                (
                                    trajectory_buffer,
                                    self._get_clip_name(clip_session_uuid, camera_name, "bin"),
                                )
                            )

                    # Upload tar archive
                    write_bytes(
                        _create_tar_bytes(samples_to_write),
                        self._get_tar_url(clip_session_uuid),
                        f"{clip_session_uuid}",
                        f"{clip_session_uuid}",
                        verbose=self._verbose,
                        client=self._s3_client_output,
                    )
                    samples_to_write.clear()
                    num_uploaded_tars += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"S3 uploading failure: {e!s}")
                task.s3_upload_error = True
            else:
                # abort if any clip within this session failed
                logger.info(f"Uploaded {num_uploaded_tars} mp4 tars from {len(task.samples)} sessions")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task


class T5EmbeddingPackagingStageE(CuratorStage):
    """Stage for packaging T5 embeddings into tar archives (Embeddings-first format).

    This stage handles packaging T5 embeddings and associated metadata into tar archives,
    organizing them by clip session with embeddings as the primary organization unit.
    """

    def __init__(  # noqa: PLR0913
        self,
        db: PostgresDB,
        camera_format_id: str,
        dataset_name: str,
        output_prefix: str,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that packages T5 embeddings into tar archives.

        Args:
            db: PostgreSQL database configuration
            camera_format_id: ID for camera format configuration
            dataset_name: Name of the dataset
            output_prefix: Base path for output files
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        """
        self._timer = StageTimer(self)
        self._camera_mapping = CAMERA_MAPPING[camera_format_id]["camera_name_mapping_cosmos"]
        self._output_prefix = output_prefix.rstrip("/")
        self._dataset_prefix = f"{db.env_type.value}/datasets/{dataset_name}"
        self._verbose = verbose
        self._log_stats = log_stats

    def stage_setup(self) -> None:
        """Set up S3 clients for input and output operations."""
        self._s3_client_input = s3_client.create_s3_client(target_path=self._output_prefix)
        self._s3_client_output = s3_client.create_s3_client(target_path=self._output_prefix)

    def _get_tar_url(
        self,
        t5_variant: str,
        clip_session_uuid: uuid.UUID,
    ) -> s3_client.S3Prefix | Path:
        """Generate a URL or path for storing T5 embedding tar archive data.

        Args:
            t5_variant: T5 model variant name
            clip_session_uuid: UUID of the clip session

        Returns:
            An S3Prefix object if the output_prefix is an S3 path, otherwise a
            pathlib.Path object

        """
        full_path = f"{self._output_prefix}/{self._dataset_prefix}/{t5_variant}/{clip_session_uuid}.tar"
        if is_s3path(self._output_prefix):
            return s3_client.S3Prefix(full_path)
        return Path(full_path)

    def _get_item_name(self, clip_session_uuid: uuid.UUID, camera_name: str, file_ext: str) -> str:
        """Generate a filename for a T5 embedding component.

        Args:
            clip_session_uuid: UUID of the clip session
            camera_name: Name of the camera
            file_ext: File extension

        Returns:
            Generated filename

        """
        return f"{clip_session_uuid}.{camera_name}.{file_ext}"

    def process_data(self, tasks: list[AvShardingTask]) -> list[AvShardingTask] | None:  # type: ignore[override]
        """Process and package T5 embeddings with performance tracking.

        This method packages T5 embeddings and associated metadata into tar archives,
        organizing them by clip session with embeddings as the primary organization unit.

        Args:
            tasks: Tasks containing T5 embeddings to package

        Returns:
            List containing the input task if successful

        Raises:
            Exception: If storage operations fail

        """
        return [self._process_task(task) for task in tasks]

    def _process_task(self, task: AvShardingTask) -> AvShardingTask:
        self._timer.reinit(self, task.get_major_size())
        # upload caption json and t5 embeddings to s3
        with self._timer.time_process():
            num_uploaded_tars = 0
            try:
                for sample in task.samples:
                    clip_session_uuid = sample.clip_session_uuid
                    if any(x not in sample.camera_ids for x in self._camera_mapping):
                        logger.warning(f"clip-session {clip_session_uuid} does not have all cameras")
                        continue
                    samples_to_write: dict[str, list[tuple[bytes, str]]] = {x: [] for x in T5_VARIANTS.values()}
                    for idx in range(len(sample.camera_ids)):
                        camera_id = sample.camera_ids[idx]
                        if camera_id not in self._camera_mapping:
                            continue
                        camera_name = self._camera_mapping[camera_id]
                        # read t5 embeddings
                        t5_buffer = read_bytes(
                            sample.t5_urls[idx],
                            self._s3_client_input,
                        )
                        t5_embeddings = pickle.loads(t5_buffer)  # noqa: S301
                        for k in range(len(t5_embeddings)):
                            t5_variant = T5_VARIANTS[k]
                            # write t5 embeddings
                            samples_to_write[t5_variant].append(
                                (
                                    pickle.dumps(t5_embeddings[k]),
                                    self._get_item_name(clip_session_uuid, camera_name, "bin"),
                                )
                            )
                            # write metadata
                            metadata = [
                                str(sample.clip_uuids[idx]),
                                [sample.window_captions[idx][k]],
                                [sample.window_start_frames[idx][k]],  # type: ignore[index]
                                [sample.window_end_frames[idx][k]],  # type: ignore[index]
                            ]
                            samples_to_write[t5_variant].append(
                                (
                                    json.dumps(metadata).encode("utf-8"),
                                    self._get_item_name(clip_session_uuid, camera_name, "json"),
                                )
                            )
                    # upload to s3
                    for t5_variant, samples in samples_to_write.items():
                        write_bytes(
                            _create_tar_bytes(samples),
                            self._get_tar_url(t5_variant, clip_session_uuid),
                            f"{clip_session_uuid}-{t5_variant}",
                            f"{clip_session_uuid}-{t5_variant}",
                            verbose=self._verbose,
                            client=self._s3_client_output,
                        )
                    samples_to_write.clear()
                    num_uploaded_tars += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"S3 uploading failure: {e!s}")
                task.s3_upload_error = True
            else:
                # abort if any clip within this session failed
                logger.info(f"Uploaded {num_uploaded_tars} t5 tars from {len(task.samples)} sessions")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task


class T5EmbeddingPackagingStageH(CuratorStage):
    """Stage for packaging T5 embeddings into tar archives (Hierarchical format).

    This stage handles packaging T5 embeddings and associated metadata into tar archives,
    organizing them hierarchically by part number and tar index. It supports multiple
    camera views and T5 embedding variants.
    """

    def __init__(  # noqa: PLR0913
        self,
        db: PostgresDB,
        camera_format_id: str,
        dataset_name: str,
        output_prefix: str,
        verbose: bool = False,  # noqa: FBT001, FBT002
        log_stats: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a stage that packages T5 embeddings into hierarchical tar archives.

        Args:
            db: PostgreSQL database configuration
            camera_format_id: ID for camera format configuration
            dataset_name: Name of the dataset
            output_prefix: Base path for output files
            verbose: Whether to enable verbose logging
            log_stats: Whether to log performance statistics

        """
        self._timer = StageTimer(self)
        self._camera_mapping = CAMERA_MAPPING[camera_format_id]["camera_name_mapping_cosmos"]
        self._camera_id_mapping_cosmos = CAMERA_MAPPING[camera_format_id]["camera_id_mapping_cosmos"]
        self._dataset_name = str(dataset_name)
        self._output_prefix = output_prefix.rstrip("/")
        self._embedding_prefix = f"{db.env_type.value}/datasets/{dataset_name}"
        self._verbose = verbose
        self._log_stats = log_stats

    def stage_setup(self) -> None:
        """Set up S3 clients for input and output operations."""
        self._s3_client_input = s3_client.create_s3_client(target_path=self._output_prefix)
        self._s3_client_output = s3_client.create_s3_client(target_path=self._output_prefix)

    def sessions_per_part(self) -> int:
        """Calculate number of sessions that can fit in one part.

        Returns:
            Number of sessions per part based on camera count and embedding size

        """
        return int(MAX_TARS_PER_PART * _EMBEDDINGS_PER_TAR / len(self._camera_mapping.keys()))

    def get_chunk_prefix(self, t5_variant: str) -> str:
        """Generate base prefix for a T5 variant's chunks.

        Args:
            t5_variant: T5 model variant name

        Returns:
            Base path prefix for the T5 variant's chunks

        """
        return f"{self._output_prefix}/{self._embedding_prefix}/{t5_variant}"

    def _get_tar_url(self, t5_variant: str, part_num: int, tar_idx: int) -> s3_client.S3Prefix | Path:
        """Generate a URL or path for storing T5 embedding tar archive data.

        Args:
            t5_variant: T5 model variant name
            part_num: Part number for hierarchical organization
            tar_idx: Index of the tar archive within the part

        Returns:
            An S3Prefix object if the output_prefix is an S3 path, otherwise a
            pathlib.Path object

        """
        full_path = f"{self.get_chunk_prefix(t5_variant)}/part_{part_num:06d}/t5_{tar_idx:06d}.tar"
        if is_s3path(self._output_prefix):
            return s3_client.S3Prefix(full_path)
        return Path(full_path)

    def _get_metadata_url(self, t5_variant: str, part_num: int, tar_idx: int) -> s3_client.S3Prefix | Path:
        """Generate a URL or path for storing T5 embedding metadata.

        Args:
            t5_variant: T5 model variant name
            part_num: Part number for hierarchical organization
            tar_idx: Index of the tar archive within the part

        Returns:
            An S3Prefix object if the output_prefix is an S3 path, otherwise a
            pathlib.Path object

        """
        full_path = f"{self.get_chunk_prefix(t5_variant)}/part_{part_num:06d}/t5_{tar_idx:06d}.json"
        if is_s3path(self._output_prefix):
            return s3_client.S3Prefix(full_path)
        return Path(full_path)

    def process_data(self, tasks: list[AvShardingTask]) -> list[AvShardingTask] | None:  # type: ignore[override]
        """Process and package T5 embeddings with performance tracking.

        This method packages T5 embeddings and associated metadata into tar archives,
        organizing them hierarchically by part number and tar index. It handles multiple
        camera views and T5 embedding variants.

        Args:
            tasks: Tasks containing T5 embeddings to package

        Returns:
            List containing the input task if successful

        Raises:
            Exception: If storage operations fail

        """
        return [self._process_task(task) for task in tasks]

    def _process_task(self, task: AvShardingTask) -> AvShardingTask:  # noqa: C901
        self._timer.reinit(self, task.get_major_size())
        # upload t5 embedding to s3
        with self._timer.time_process():
            num_uploaded_tars = 0
            try:
                samples_to_write = {x: [] for x in range(WINDOWS_PER_CLIP)}  # type: ignore[var-annotated]
                metadatas = {x: {} for x in range(WINDOWS_PER_CLIP)}  # type: ignore[var-annotated]
                task.tar_mappings = {x: {} for x in range(WINDOWS_PER_CLIP)}
                for sample in task.samples:
                    clip_session_uuid = str(sample.clip_session_uuid)
                    if any(x not in sample.camera_ids for x in self._camera_mapping):
                        logger.warning(f"clip-session {clip_session_uuid} does not have all cameras")
                        continue
                    for idx in range(len(sample.camera_ids)):
                        camera_id = sample.camera_ids[idx]
                        if camera_id not in self._camera_mapping:
                            continue
                        camera_name = self._camera_mapping[camera_id]
                        cosmos_camera_id = self._camera_id_mapping_cosmos[camera_id]
                        # read t5 embeddings
                        t5_buffer = read_bytes(
                            sample.t5_urls[idx],
                            self._s3_client_input,
                        )
                        t5_embeddings = pickle.loads(t5_buffer)  # noqa: S301
                        if len(t5_embeddings) != len(sample.window_captions[idx]):
                            logger.error(
                                f"clip-session {clip_session_uuid} camera-{camera_id} "
                                f"only has {len(t5_embeddings)} t5 embeddings"
                            )
                            task.source_data_error = True
                            continue
                        filename = f"{clip_session_uuid}.{camera_name}"
                        for k in range(len(t5_embeddings)):
                            samples_to_write[k].append((pickle.dumps(t5_embeddings[k]), f"{filename}.bin"))
                            if clip_session_uuid not in metadatas[k]:
                                metadatas[k][clip_session_uuid] = {}
                            metadatas[k][clip_session_uuid][cosmos_camera_id] = [
                                self._dataset_name,
                                [sample.window_captions[idx][k]],
                                [sample.window_start_frames[idx][k]],  # type: ignore[index]
                                [sample.window_end_frames[idx][k]],  # type: ignore[index]
                            ]
                        if len(samples_to_write[0]) == _EMBEDDINGS_PER_TAR:
                            self._upload_one_tar(
                                task,
                                num_uploaded_tars,
                                samples_to_write,
                                metadatas,
                            )
                            num_uploaded_tars += 1
                if len(samples_to_write[0]) > 0:
                    self._upload_one_tar(task, num_uploaded_tars, samples_to_write, metadatas)
                    num_uploaded_tars += 1
            except Exception as e:  # noqa: BLE001
                logger.error(f"S3 uploading failure: {e!s}")
                task.s3_upload_error = True
            else:
                # abort if any clip within this session failed
                logger.info(f"Uploaded {num_uploaded_tars} tars from {len(task.samples)} samples")

        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            task.stage_perf[stage_name] = stage_perf_stats
        return task

    def _upload_one_tar(
        self,
        task: AvShardingTask,
        tar_idx: int,
        samples_to_write: dict[int, list[tuple[bytes, str]]],
        metadatas: dict[int, dict[str, dict[str, Any]]],
    ) -> None:
        """Upload a single tar archive and its metadata.

        Args:
            task: Task containing packaging information
            tar_idx: Index of the tar archive
            samples_to_write: Dictionary mapping window indices to lists of (data, filename) tuples
            metadatas: Dictionary mapping window indices to metadata dictionaries

        """
        for k, tar_samples in samples_to_write.items():
            url = self._get_tar_url(T5_VARIANTS[k], task.part_num, tar_idx)
            write_bytes(
                _create_tar_bytes(tar_samples),
                url,
                f"t5_{tar_idx:06d}",
                f"part_{task.part_num:06d}",
                verbose=self._verbose,
                client=self._s3_client_output,
            )

            # create bin -> tar mapping
            for _, filename in tar_samples:
                task.tar_mappings[k][filename.removesuffix(".bin")] = str(url)

        for k, json_metadata in metadatas.items():
            url = self._get_metadata_url(T5_VARIANTS[k], task.part_num, tar_idx)
            write_json(
                json_metadata,
                url,
                f"t5_{tar_idx:06d}",
                f"part_{task.part_num:06d}",
                verbose=self._verbose,
                client=self._s3_client_output,
            )

        # clear
        for k in samples_to_write:  # noqa: PLC0206
            samples_to_write[k].clear()
        for k in metadatas:  # noqa: PLC0206
            metadatas[k].clear()
