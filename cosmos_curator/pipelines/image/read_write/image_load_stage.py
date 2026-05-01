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

"""Image load stage: read image from path or S3 into task encoded_data."""

import pathlib

import numpy as np
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.image_sensor import ImageSensor
from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.storage import storage_client, storage_utils
from cosmos_curator.pipelines.image.utils.data_model import Image, ImagePipeTask
from cosmos_curator.pipelines.image.utils.image_pipe_input import get_image_relative_paths


class ImageLoadStage(CuratorStage):
    """Stage that loads image files from storage (local or S3) into task encoded_data."""

    def __init__(
        self,
        input_path: str,
        input_s3_profile_name: str,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the image load stage.

        Args:
            input_path: Base path for input images (used to create storage client).
            input_s3_profile_name: S3 profile name for remote input.
            verbose: Whether to log per-image load.
            log_stats: Whether to record stage performance in task.stage_perf.

        """
        self._timer = StageTimer(self)
        self._input_path = input_path
        self._input_s3_profile_name = input_s3_profile_name
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: storage_client.StorageClient | None = None
        self._image_sensor: ImageSensor | None = None
        self._start_ns_by_relative_path: dict[str, int] = {}

    @property
    def resources(self) -> CuratorStageResource:
        """Resource configuration for this stage."""
        return CuratorStageResource(cpus=0.25)

    def stage_setup(self) -> None:
        """Initialize storage client."""
        self._client = storage_utils.get_storage_client(self._input_path, profile_name=self._input_s3_profile_name)
        self._ensure_image_sensor()

    def _ensure_image_sensor(self) -> None:
        """Initialize a shared collection-level ``ImageSensor`` for this worker."""
        if self._image_sensor is not None:
            return

        relative_paths = get_image_relative_paths(self._input_path, self._input_s3_profile_name)
        if not relative_paths:
            msg = f"No input images found under {self._input_path}"
            raise ValueError(msg)

        raw_sources = [storage_utils.get_full_path(self._input_path, rel) for rel in relative_paths]
        sources = [s if isinstance(s, pathlib.Path) else str(s) for s in raw_sources]
        client_params = storage_utils.get_smart_open_client_params(self._client) if self._client is not None else None
        sensor_timestamps_ns = np.arange(len(sources), dtype=np.int64)
        self._image_sensor = ImageSensor(
            sources,
            sensor_timestamps_ns=sensor_timestamps_ns,
            client_params=client_params,
        )
        self._start_ns_by_relative_path = {
            rel: int(sensor_timestamps_ns[idx]) for idx, rel in enumerate(relative_paths)
        }

    def _load_image_bytes(self, image: Image) -> bool:
        """Load image bytes from path or S3 into image.encoded_data.

        Returns True on success, False on error (and sets image.errors["download"]).
        """
        try:
            if isinstance(image.input_image, pathlib.Path):
                with image.input_image.open("rb") as fp:
                    image.encoded_data = bytes_to_numpy(fp.read())  # type: ignore[assignment]
            elif self._client is not None:
                image.encoded_data = bytes_to_numpy(  # type: ignore[assignment]
                    storage_utils.read_bytes(image.input_image, self._client)
                )
            else:
                raise ValueError("S3 client is required for S3 input")  # noqa: TRY301, EM101
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to read image {image.input_image}: {e}")
            image.errors["download"] = str(e)
            return False

        if not image.encoded_data or image.encoded_data.nbytes == 0:
            logger.error(f"Empty or missing encoded_data for {image.input_image}")
            image.errors["download"] = "empty file"
            return False

        return True

    def _load_image_data(self, image: Image) -> bool:
        """Decode sampled image data for downstream captioning stages."""
        self._ensure_image_sensor()
        sensor = self._image_sensor
        if sensor is None:
            image.errors["decode"] = "shared ImageSensor was not initialized"
            logger.error(f"Failed to decode image {image.input_image}: {image.errors['decode']}")
            return False

        start_ns = self._start_ns_by_relative_path.get(image.relative_path)
        if start_ns is None:
            image.errors["decode"] = (
                f"image relative_path {image.relative_path!r} was not found in the shared ImageSensor index"
            )
            logger.error(f"Failed to decode image {image.input_image}: {image.errors['decode']}")
            return False

        try:
            grid = SamplingGrid(
                start_ns=start_ns,
                exclusive_end_ns=start_ns + 1,
                timestamps_ns=np.array([start_ns], dtype=np.int64),
                stride_ns=1,
                duration_ns=1,
            )
            image.image_data = next(sensor.sample(SamplingSpec(grid=grid)))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to decode image {image.input_image}: {e}")
            image.errors["decode"] = str(e)
            return False

        image.width = image.image_data.metadata.width
        image.height = image.image_data.metadata.height
        return True

    @nvtx.annotate("ImageLoadStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Load each task's image from storage into task.image.encoded_data."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            with self._timer.time_process():
                loaded_bytes = self._load_image_bytes(task.image)
                loaded_image_data = loaded_bytes and self._load_image_data(task.image)
                if loaded_image_data and self._verbose:
                    logger.info(f"Loaded image {task.image.input_image} size={task.image.encoded_data.nbytes:,}B")
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
