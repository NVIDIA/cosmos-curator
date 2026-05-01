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
"""Video download and inline remux stages."""

import json
import pathlib
import pickle
import subprocess

import numpy as np
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.dataset import webdataset_utils
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.storage import storage_client, storage_utils
from cosmos_curator.core.utils.storage.writer_utils import write_bytes
from cosmos_curator.pipelines.video.read_write.remux_stages import remux_if_needed
from cosmos_curator.pipelines.video.utils.data_model import (
    ShardPipeTask,
    SplitPipeTask,
    Video,
)


def _value_error(msg: str) -> None:
    raise ValueError(msg)


class VideoDownloader(CuratorStage):
    """Stage that downloads video files from storage and extracts metadata.

    This class processes video files through a series of steps including downloading,
    extracting metadata, and storing the results in the task.
    """

    def __init__(
        self,
        input_path: str,
        input_s3_profile_name: str,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the video downloader stage.

        Args:
            input_path: Path to input videos.
            input_s3_profile_name: S3 profile name for input.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._input_path = input_path
        self._input_s3_profile_name = input_s3_profile_name
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: storage_client.StorageClient | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(cpus=0.25)

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        self._client = storage_utils.get_storage_client(self._input_path, profile_name=self._input_s3_profile_name)

    def _download_video_bytes(self, video: Video) -> bool:
        """Download video bytes from storage.

        Args:
            video: Video object to download bytes for.

        Returns:
            True if download successful, False otherwise.

        """
        try:
            if isinstance(video.input_video, pathlib.Path):
                with video.input_video.open("rb") as fp:
                    video.encoded_data = bytes_to_numpy(fp.read())  # type: ignore[assignment]
            elif self._client is not None:
                video.encoded_data = bytes_to_numpy(storage_utils.read_bytes(video.input_video, self._client))  # type: ignore[assignment]
            else:
                _value_error("S3 client is required for S3 destination")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Got an exception {e!s} when trying to read {video.input_video}")
            video.errors["download"] = str(e)
            return False

        if not video.encoded_data:
            # should never happen, but log it just in case
            logger.error(f"video.encoded_data is None for {video.input_video} without exceptions ???")
            video.encoded_data = np.array([], dtype=np.uint8)  # type: ignore[assignment]

        return True

    def _extract_and_validate_metadata(self, video: Video) -> bool:
        """Extract metadata and validate video properties.

        Args:
            video: Video object to extract metadata for.

        Returns:
            True if metadata extraction successful, False otherwise.

        """
        try:
            video.populate_metadata()
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode(errors="replace").strip() if e.stderr else ""
            logger.warning(
                f"Failed to extract metadata for {video.input_video}: {e}"
                + (f"\n  stderr: {stderr_text}" if stderr_text else "")
            )
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to extract metadata for {video.input_video}: {e}")
            return False

        # Log warnings for missing metadata
        if video.metadata.video_codec is None:
            logger.warning(f"Codec could not be extracted for {video.input_video}!")
        if video.metadata.pixel_format is None:
            logger.warning(f"Pixel format could not be extracted for {video.input_video}!")

        return True

    def _format_metadata_for_logging(self, video: Video) -> dict[str, str]:
        """Format video metadata for logging, handling None values gracefully.

        Args:
            video: Video object with metadata.

        Returns:
            Dictionary of formatted metadata strings.

        """
        metadata = video.metadata

        # Format each field, using "unknown" for None values
        return {
            "size": f"{video.encoded_data.nbytes:,}B" if video.encoded_data.nbytes else "0B",
            "res": f"{metadata.width or 'unknown'}x{metadata.height or 'unknown'}",
            "fps": f"{metadata.framerate:.1f}" if metadata.framerate is not None else "unknown",
            "duration": f"{metadata.duration / 60:.0f}m" if metadata.duration is not None else "unknown",
            "weight": f"{video.weight:.2f}" if metadata.duration is not None else "unknown",
            "bit_rate": f"{metadata.bit_rate_k}K" if metadata.bit_rate_k is not None else "unknown",
        }

    def _log_video_info(self, video: Video) -> None:
        """Log video information after successful download and metadata extraction.

        Args:
            video: Video object with metadata.

        """
        meta = self._format_metadata_for_logging(video)
        logger.info(
            f"Downloaded {video.input_video} "
            f"size={meta['size']} "
            f"res={meta['res']} "
            f"fps={meta['fps']} "
            f"duration={meta['duration']} "
            f"weight={meta['weight']} "
            f"bit_rate={meta['bit_rate']}.",
        )

    @nvtx.annotate("VideoDownloader")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Read video(s) specified in URI to task buffer."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            for video in task.videos:
                with self._timer.time_process():
                    # Download video bytes
                    if not self._download_video_bytes(video):
                        continue

                    # Extract and validate metadata
                    if not self._extract_and_validate_metadata(video):
                        continue

                    try:
                        video.was_remuxed = remux_if_needed(video, threads=1)
                    except Exception as e:  # noqa: BLE001
                        video.errors["remux"] = str(e)
                        logger.exception(f"Failed to remux video {video.input_video}")
                        continue

                    # TODO(LazyData): re-enable when batch-mode ObjectRef ownership is
                    # resolved.  In batch mode, pool.stop() kills actor -> OwnerDiedError.
                    # video.encoded_data.store()  # noqa: ERA001

                    try:
                        video.populate_timestamps()
                    except Exception as e:  # noqa: BLE001
                        video.errors["timestamps"] = str(e)  # authoritative setter; intentional overwrite on retry
                        logger.exception(f"Failed to populate timestamps for {video.input_video}")

                    # Log video information
                    self._log_video_info(video)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class DownloadPackUpload(CuratorStage):
    """Stage that downloads video clips and packs them into a webdataset.

    This class processes video clips through a series of steps including downloading,
    packing into a webdataset, and writing to storage.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_path: str,
        output_path: str,
        input_s3_profile_name: str,
        output_s3_profile_name: str,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the download pack upload stage.

        Args:
            input_path: Path to input videos.
            output_path: Path to write output files.
            input_s3_profile_name: S3 profile name for input.
            output_s3_profile_name: S3 profile name for output.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._input_path = input_path
        self._output_path = output_path
        self._input_s3_profile_name = input_s3_profile_name
        self._output_s3_profile_name = output_s3_profile_name
        self._verbose = verbose
        self._log_stats = log_stats
        self._client_input: storage_client.StorageClient | None = None
        self._client_output: storage_client.StorageClient | None = None

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        self._client_input = storage_utils.get_storage_client(
            self._input_path,
            profile_name=self._input_s3_profile_name,
        )
        self._client_output = storage_utils.get_storage_client(
            self._output_path,
            profile_name=self._output_s3_profile_name,
        )

    def _download_clips(self, task: ShardPipeTask) -> None:
        for clip in task.samples:
            with self._timer.time_process():
                try:
                    if isinstance(clip.clip_location, pathlib.Path):
                        with clip.clip_location.open("rb") as fp:
                            clip.encoded_data = bytes_to_numpy(fp.read())  # type: ignore[assignment]
                    elif self._client_input is not None:
                        clip.encoded_data = bytes_to_numpy(  # type: ignore[assignment]
                            storage_utils.read_bytes(clip.clip_location, self._client_input),
                        )
                    else:
                        _value_error("S3 client is required for S3 destination")

                except Exception as e:
                    error_msg = f"Got an exception {e!s} when trying to read {clip.clip_location}"
                    raise RuntimeError(error_msg) from e

                else:
                    if self._verbose:
                        logger.info(f"Downloaded {clip.clip_location}: size={clip.encoded_data.nbytes:,}Byte")

    def _write_tar(
        self,
        tar_bytes: bytes,
        output_path: storage_client.StoragePrefix | pathlib.Path,
        key: str,
        part_num: int,
    ) -> None:
        write_bytes(
            tar_bytes,
            output_path,
            f"tar {key}",
            str(part_num),
            verbose=self._verbose,
            client=self._client_output,
        )

    @nvtx.annotate("DownloadPackUpload")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ShardPipeTask]) -> list[ShardPipeTask] | None:
        """Read video specified in URI to task buffer."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            # first, download all clips
            self._download_clips(task)

            # then, pack them into a webdataset
            samples_to_write_mp4: list[webdataset_utils.RawSample] = []
            samples_to_write_metas: list[webdataset_utils.RawSample] = []
            samples_to_write_t5_xxls: list[webdataset_utils.RawSample] = []
            for clip in task.samples:
                clip_mp4_data = clip.encoded_data.resolve()
                mp4_bytes = clip_mp4_data.tobytes() if clip_mp4_data is not None else b""
                samples_to_write_mp4.append(webdataset_utils.RawSample(clip.uuid, {"mp4": mp4_bytes}))
                samples_to_write_t5_xxls.append(
                    webdataset_utils.RawSample(clip.uuid, {"pickle": pickle.dumps(clip.t5_xxl_embeddings)}),
                )
                samples_to_write_metas.append(
                    webdataset_utils.RawSample(clip.uuid, {"json": json.dumps(clip.clip_metadata)}),
                )
                task.key_count += 1
            tar_bytes_mp4 = webdataset_utils.make_tar_from_samples(samples_to_write_mp4)
            tar_bytes_t5_xxls = webdataset_utils.make_tar_from_samples(samples_to_write_t5_xxls)
            tar_bytes_metas = webdataset_utils.make_tar_from_samples(samples_to_write_metas)
            self._write_tar(tar_bytes_mp4, task.output_tar_video, "video", task.part_num)
            self._write_tar(tar_bytes_t5_xxls, task.output_tar_t5_xxl, "t5_xxl", task.part_num)
            self._write_tar(tar_bytes_metas, task.output_tar_metas, "metas", task.part_num)

            # clear intermediate buffers
            for clip in task.samples:
                clip.clip_metadata.clear()
                clip.encoded_data.drop()
                clip.t5_xxl_embeddings.clear()

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
