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
"""Embedding stages for the image pipeline."""

import concurrent.futures
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curator.core.interfaces.model_interface import ModelInterface
from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.config.config import maybe_load_config, resolve_model_name_auto
from cosmos_curator.core.utils.infra.gpu_start_helper import gpu_stage_cleanup, gpu_stage_startup
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.model import conda_utils
from cosmos_curator.models.clip import CLIPImageEmbeddings
from cosmos_curator.models.cosmos_embed1 import CosmosEmbed1
from cosmos_curator.models.internvideo2_mm import InternVideo2MultiModality
from cosmos_curator.pipelines.common.openai_embedding_utils import call_openai_embedding_api, frame_to_base64_jpeg
from cosmos_curator.pipelines.image.utils.data_model import ImagePipeTask

if TYPE_CHECKING:
    import openai

if conda_utils.is_running_in_env("unified"):
    import openai


class ImageCosmosEmbed1EmbeddingStage(CuratorStage):
    """Embed images using Cosmos-Embed1 by replicating the frame to satisfy the video frame count."""

    def __init__(
        self,
        variant: Literal["224p", "336p", "448p"] = "336p",
        num_gpus_per_worker: float = 0.25,
        *,
        verbose: bool = False,
        log_stats: bool = False,
        texts_to_verify: list[str] | None = None,
    ) -> None:
        """Initialize the Cosmos-Embed1 image embedding stage.

        Args:
            variant: Cosmos-Embed1 model variant ("224p", "336p", or "448p").
            num_gpus_per_worker: Number of GPUs per worker.
            verbose: Whether to emit verbose per-image logs.
            log_stats: Whether to record stage performance in task.stage_perf.
            texts_to_verify: Optional list of texts to verify against embeddings.

        """
        self._timer = StageTimer(self)
        self._variant = variant
        self._num_gpus_per_worker = num_gpus_per_worker
        self._verbose = verbose
        self._log_stats = log_stats
        self._texts_to_verify = texts_to_verify
        self._model = CosmosEmbed1(variant=variant, utils_only=False)
        self._process_count = 0

    @property
    def model(self) -> ModelInterface:
        """Return the Cosmos-Embed1 model interface."""
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Return the GPU resource requirements for this stage."""
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    def stage_setup(self) -> None:
        """Initialize GPU resources and load model weights."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Release the GPU-resident Cosmos-Embed1 model before the actor exits.

        Drops ``self._model`` first so its CUDA weights become unreachable, then runs
        the standard ``gpu_stage_cleanup`` to return device memory to the driver. See
        ``InternVideo2EmbeddingStage.destroy`` for the rationale.
        """
        self._model = None  # type: ignore[assignment]
        gpu_stage_cleanup(self.__class__.__name__)

    @nvtx.annotate("ImageCosmosEmbed1EmbeddingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Embed each image using Cosmos-Embed1's single-image wrapper path."""
        model_key = f"cosmos_embed1_{self._variant}"
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            image = task.image
            if image.image_data is None or len(image.image_data.frames) == 0:
                image.errors[model_key] = "no image_data"
                continue
            with self._timer.time_process():
                frame: npt.NDArray[np.uint8] = image.image_data.frames[0]
                ce1_input = self._model.formulate_input_image(frame)
                embedding = self._model.encode_video_frames(ce1_input)
                if embedding.numel() == 0:
                    image.errors[model_key] = "encode failed"
                    logger.error(f"Cosmos-Embed1 encode_video_frames returned empty for {task.session_id}")
                    continue
                image.embeddings[model_key] = embedding.cpu().numpy()
                if self._texts_to_verify:
                    text_embeddings = [self._model.get_text_embedding(x) for x in self._texts_to_verify]
                    probs, idxs = self._model.evaluate(embedding, text_embeddings)
                    image.cosmos_embed1_text_match = (self._texts_to_verify[idxs[0]], probs[0])
                if self._verbose:
                    logger.info(f"Cosmos-Embed1 embedded {task.session_id}: shape={image.embeddings[model_key].shape}")
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        self._process_count += 1
        if self._process_count % 10 == 0:
            torch.cuda.empty_cache()

        return tasks


class ImageInternVideo2EmbeddingStage(CuratorStage):
    """Embed images using InternVideo2's single-image wrapper path."""

    def __init__(
        self,
        num_gpus_per_worker: float = 0.25,
        *,
        verbose: bool = False,
        log_stats: bool = False,
        texts_to_verify: list[str] | None = None,
    ) -> None:
        """Initialize the InternVideo2 image embedding stage.

        Args:
            num_gpus_per_worker: Number of GPUs per worker.
            verbose: Whether to emit verbose per-image logs.
            log_stats: Whether to record stage performance in task.stage_perf.
            texts_to_verify: Optional list of texts to verify against embeddings.

        """
        self._timer = StageTimer(self)
        self._num_gpus_per_worker = num_gpus_per_worker
        self._verbose = verbose
        self._log_stats = log_stats
        self._texts_to_verify = texts_to_verify
        self._model = InternVideo2MultiModality(utils_only=False)
        self._process_count = 0

    @property
    def model(self) -> ModelInterface:
        """Return the InternVideo2 model interface."""
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Return the GPU resource requirements for this stage."""
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    def stage_setup(self) -> None:
        """Initialize GPU resources and load model weights."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Release the GPU-resident InternVideo2 model before the actor exits.

        Drops ``self._model`` first so its CUDA weights become unreachable, then runs
        the standard ``gpu_stage_cleanup`` to return device memory to the driver. See
        ``InternVideo2EmbeddingStage.destroy`` (in the video embedding stages) for the
        full rationale.
        """
        self._model = None  # type: ignore[assignment]
        gpu_stage_cleanup(self.__class__.__name__)

    @nvtx.annotate("ImageInternVideo2EmbeddingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Embed each image using InternVideo2's native image-aware model path."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            image = task.image
            if image.image_data is None or len(image.image_data.frames) == 0:
                image.errors["internvideo2_embedding"] = "no image_data"
                continue
            with self._timer.time_process():
                frame: npt.NDArray[np.uint8] = image.image_data.frames[0]
                iv2_input = self._model.formulate_input_image(frame)
                if iv2_input.size == 0:
                    image.errors["internvideo2_embedding"] = "formulate failed"
                    logger.error(f"InternVideo2 formulate_input_image failed for {task.session_id}")
                    continue
                embedding_tensor = self._model.encode_video_frames(iv2_input)
                image.embeddings["internvideo2"] = embedding_tensor.cpu().float().numpy()
                if self._texts_to_verify:
                    text_embeddings = [self._model.get_text_embedding(x) for x in self._texts_to_verify]
                    probs, idxs = self._model.evaluate(
                        torch.from_numpy(image.embeddings["internvideo2"]), text_embeddings
                    )
                    image.intern_video_2_text_match = (self._texts_to_verify[idxs[0]], probs[0])
                if self._verbose:
                    logger.info(
                        f"InternVideo2 embedded {task.session_id}: shape={image.embeddings['internvideo2'].shape}"
                    )
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        self._process_count += 1
        if self._process_count % 10 == 0:
            torch.cuda.empty_cache()

        return tasks


class ImageCLIPEmbeddingStage(CuratorStage):
    """Embed images using CLIP (openai/clip-vit-large-patch14)."""

    def __init__(
        self,
        num_gpus_per_worker: float = 0.25,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the CLIP image embedding stage.

        Args:
            num_gpus_per_worker: Number of GPUs per worker.
            verbose: Whether to emit verbose per-image logs.
            log_stats: Whether to record stage performance in task.stage_perf.

        """
        self._timer = StageTimer(self)
        self._num_gpus_per_worker = num_gpus_per_worker
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = CLIPImageEmbeddings()

    @property
    def model(self) -> ModelInterface:
        """Return the CLIP model interface."""
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Return the GPU resource requirements for this stage."""
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    def stage_setup(self) -> None:
        """Initialize GPU resources and load model weights."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Release the GPU-resident CLIP image embedding model before the actor exits.

        Drops ``self._model`` first so its CUDA weights become unreachable, then runs
        the standard ``gpu_stage_cleanup`` to return device memory to the driver. See
        ``InternVideo2EmbeddingStage.destroy`` (in the video embedding stages) for the
        full rationale.
        """
        self._model = None  # type: ignore[assignment]
        gpu_stage_cleanup(self.__class__.__name__)

    @nvtx.annotate("ImageCLIPEmbeddingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Embed each image using CLIP."""
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            image = task.image
            if image.image_data is None or len(image.image_data.frames) == 0:
                image.errors["clip_embedding"] = "no image_data"
                continue
            with self._timer.time_process():
                frame: npt.NDArray[np.uint8] = image.image_data.frames[0]
                batch = frame[np.newaxis, ...]  # (1, H, W, 3)
                embedding_tensor = self._model(batch)
                image.embeddings["clip"] = embedding_tensor[0].cpu().numpy()
                if self._verbose:
                    logger.info(f"CLIP embedded {task.session_id}: shape={image.embeddings['clip'].shape}")
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class ImageOpenAIEmbeddingStage(CuratorStage):
    """Embed images using an OpenAI-compatible embedding API (e.g. vLLM serving)."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        max_concurrent_requests: int = 8,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the OpenAI-compatible image embedding stage.

        Args:
            model_name: Model name to pass in the API request (use "auto" to detect from endpoint).
            max_retries: Number of retries per image before giving up.
            retry_delay_seconds: Delay between retries.
            max_concurrent_requests: Maximum concurrent requests to the embedding endpoint.
            verbose: Whether to emit verbose per-image logs.
            log_stats: Whether to record stage performance in task.stage_perf.

        """
        self._timer = StageTimer(self)
        self._model_name = model_name
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._max_concurrent_requests = max_concurrent_requests
        self._verbose = verbose
        self._log_stats = log_stats
        self._client: openai.OpenAI | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Return CPU resource requirements (API calls are remote)."""
        return CuratorStageResource(cpus=1.0)

    @property
    def conda_env_name(self) -> str:
        """Return the conda environment name (openai package lives in unified)."""
        return "unified"

    def stage_setup(self) -> None:
        """Create the OpenAI API client using credentials from the config file."""
        config = maybe_load_config()
        endpoint = config.openai.embedding if config is not None and config.openai is not None else None
        if endpoint is None or not endpoint.api_key:
            msg = (
                "OpenAI embedding configuration not found. "
                "Provide openai.embedding.api_key in ~/.config/cosmos_curator/config.yaml"
            )
            raise RuntimeError(msg)
        client_kwargs: dict[str, Any] = {"api_key": endpoint.api_key}
        if endpoint.base_url:
            client_kwargs["base_url"] = endpoint.base_url
        self._client = openai.OpenAI(**client_kwargs)
        self._model_name = resolve_model_name_auto(self._client, self._model_name, endpoint_label="OpenAI embedding")

    def _embed_image(self, task: ImagePipeTask) -> None:
        """Embed a single image — designed to run inside a thread pool."""
        client = self._client
        if client is None:
            task.image.errors["openai_embedding"] = "client not initialized"
            return
        image_data = task.image.image_data
        if image_data is None or len(image_data.frames) == 0:
            task.image.errors["openai_embedding"] = "no image_data"
            return
        frame: npt.NDArray[np.uint8] = image_data.frames[0]
        url = frame_to_base64_jpeg(frame)
        content_parts: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": url}}]
        try:
            embedding = call_openai_embedding_api(
                client,
                self._model_name,
                content_parts,
                max_retries=self._max_retries,
                retry_delay_seconds=self._retry_delay_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            task.image.errors["openai_embedding"] = str(exc)
            if self._verbose:
                logger.exception(f"OpenAI embedding failed for {task.session_id}")
            else:
                logger.warning(f"OpenAI embedding failed for {task.session_id}: {exc}")
            return
        task.image.embeddings["openai"] = embedding
        if self._verbose:
            logger.info(f"OpenAI embedded {task.session_id}: shape={embedding.shape}")

    @nvtx.annotate("ImageOpenAIEmbeddingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[ImagePipeTask]) -> list[ImagePipeTask] | None:
        """Embed each image by sending it to the OpenAI-compatible embedding endpoint."""
        self._timer.reinit(self, sum(task.get_major_size() for task in tasks))
        with (
            self._timer.time_process(len(tasks)),
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_concurrent_requests) as pool,
        ):
            list(pool.map(self._embed_image, tasks))
        if self._log_stats:
            stage_name, stage_perf_stats = self._timer.log_stats()
            for task in tasks:
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
